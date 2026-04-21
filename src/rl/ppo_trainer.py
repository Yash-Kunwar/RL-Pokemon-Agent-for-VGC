import os
import sys
import time
import asyncio
import torch
import numpy as np
from typing import List, Optional
from poke_env import LocalhostServerConfiguration
from poke_env.player import RandomPlayer

from src.models.transformer_policy import VGCPolicyNetwork
from src.rl.rollout_buffer import RolloutBuffer
from src.rl.reward_shaper import RewardShaper
from src.rl.ppo_updater import PPOUpdater
from src.rl.ppo_player import PPOPlayer
from src.agents.transformer_player import TransformerPlayer
from src.utils.observation import get_observation_size


# ─── PPO Config ───────────────────────────────────────────────────────────────

class PPOConfig:
    # Environment
    BATTLE_FORMAT = 'gen9randomdoublesbattle'
    SERVER_CONFIG = LocalhostServerConfiguration

    # Rollout
    BUFFER_SIZE = 2048          # steps per rollout
    BATTLES_PER_ROLLOUT = 10    # battles to collect per update

    # PPO hyperparameters
    LEARNING_RATE = 1e-4        # lower than BC to avoid forgetting
    CLIP_EPSILON = 0.2
    VALUE_COEF = 0.5
    ENTROPY_COEF = 0.01
    MAX_GRAD_NORM = 0.5
    N_EPOCHS = 4                # PPO epochs per rollout
    BATCH_SIZE = 256
    GAMMA = 0.99
    GAE_LAMBDA = 0.95

    # Training schedule
    TOTAL_UPDATES = 500         # number of PPO updates
    DENSE_REWARD_DECAY_STEPS = 300  # updates over which to decay dense rewards

    # Opponent pool
    OPPONENT_POOL_SIZE = 5      # keep last N checkpoints as opponents
    UPDATE_OPPONENT_EVERY = 50  # save current weights as new opponent every N updates

    # Evaluation
    EVAL_EVERY = 25             # evaluate every N updates
    EVAL_BATTLES = 20           # battles per evaluation

    # Checkpointing
    CHECKPOINT_DIR = 'logs/checkpoints/rl'
    SAVE_EVERY = 25

    # BC checkpoint to start from
    BC_CHECKPOINT = 'bc_best.pt'

    RESUME_FROM = None  # set to checkpoint path to resume



# ─── Opponent Pool ────────────────────────────────────────────────────────────

class OpponentPool:
    """
    Manages a pool of past model checkpoints for self-play.
    Prevents the agent from overfitting to beating one specific opponent.
    """

    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.checkpoints: List[dict] = []  # list of state_dicts

    def add(self, state_dict: dict):
        """Add current model weights to pool."""
        import copy
        self.checkpoints.append(copy.deepcopy(state_dict))
        if len(self.checkpoints) > self.max_size:
            self.checkpoints.pop(0)  # remove oldest

    def sample(self) -> dict:
        """Sample a random checkpoint from the pool."""
        idx = np.random.randint(0, len(self.checkpoints))
        return self.checkpoints[idx]

    def __len__(self):
        return len(self.checkpoints)


# ─── PPO Trainer ──────────────────────────────────────────────────────────────

class PPOTrainer:
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

        # ── Load BC checkpoint ────────────────────────────────────────────────
        print(f"Loading BC checkpoint: {config.BC_CHECKPOINT}")
        self.model = VGCPolicyNetwork().to(self.device)
        bc_checkpoint = torch.load(config.BC_CHECKPOINT, map_location=self.device)
        self.model.load_state_dict(bc_checkpoint['model_state_dict'])
        print(f"  Loaded from epoch {bc_checkpoint['epoch']}")

        # ── RL components ─────────────────────────────────────────────────────
        self.buffer = RolloutBuffer(
            buffer_size=config.BUFFER_SIZE,
            obs_dim=get_observation_size(),
        )
        self.reward_shaper = RewardShaper(dense_weight=1.0)
        self.updater = PPOUpdater(
            model=self.model,
            learning_rate=config.LEARNING_RATE,
            clip_epsilon=config.CLIP_EPSILON,
            value_coef=config.VALUE_COEF,
            entropy_coef=config.ENTROPY_COEF,
            max_grad_norm=config.MAX_GRAD_NORM,
            n_epochs=config.N_EPOCHS,
            batch_size=config.BATCH_SIZE,
            device=self.device,
        )

        # ── Opponent pool — seed with BC weights ──────────────────────────────
        self.opponent_pool = OpponentPool(max_size=config.OPPONENT_POOL_SIZE)
        self.opponent_pool.add(self.model.state_dict())

        # ── Training state ────────────────────────────────────────────────────
        self.update_count = 0
        self.total_steps = 0
        self.history = []
        self.best_win_rate = 0.0

        # ── Resume from checkpoint if specified ───────────────────────────────
        if config.RESUME_FROM and os.path.exists(config.RESUME_FROM):
            print(f"Resuming from: {config.RESUME_FROM}")
            rl_ckpt = torch.load(config.RESUME_FROM, map_location=self.device)
            self.model.load_state_dict(rl_ckpt['model_state_dict'])
            self.updater.optimizer.load_state_dict(rl_ckpt['optimizer_state_dict'])
            self.update_count = rl_ckpt['update']
            self.total_steps = rl_ckpt['total_steps']
            self.history = rl_ckpt.get('history', [])
            self.best_win_rate = max(
                (e.get('eval', {}) or {}).get('vs_random', 0)
                for e in self.history
            ) if self.history else 0.0
            print(f"  Resumed from update {self.update_count}, "
                  f"steps={self.total_steps:,}, "
                  f"best_win_rate={self.best_win_rate:.1%}")
            # Seed opponent pool with resumed weights
            self.opponent_pool.add(self.model.state_dict())

        # ── Create persistent players (avoids challenge loop issues) ──────────
        self.ppo_player = PPOPlayer(
            model=self.model,
            buffer=self.buffer,
            updater=self.updater,
            reward_shaper=RewardShaper(dense_weight=1.0),
            device=self.device,
            battle_format=config.BATTLE_FORMAT,
            server_configuration=config.SERVER_CONFIG,
            max_concurrent_battles=1,
            log_level=25,
        )
        self.opponent = self._make_opponent()

    def _get_dense_weight(self) -> float:
        """Linearly decay dense reward weight to 0."""
        decay = self.config.DENSE_REWARD_DECAY_STEPS
        return max(0.0, 1.0 - (self.update_count / decay))

    def _make_ppo_player(self) -> PPOPlayer:
        """Create a fresh PPOPlayer with current model."""
        return PPOPlayer(
            model=self.model,
            buffer=self.buffer,
            updater=self.updater,
            reward_shaper=RewardShaper(dense_weight=self._get_dense_weight()),
            device=self.device,
            battle_format=self.config.BATTLE_FORMAT,
            server_configuration=self.config.SERVER_CONFIG,
            max_concurrent_battles=1,
            log_level=25,
        )

    def _make_opponent(self) -> TransformerPlayer:
        """Create an opponent from the pool."""
        opponent_model = VGCPolicyNetwork().to(self.device)
        opponent_model.load_state_dict(self.opponent_pool.sample())
        opponent_model.eval()
        return TransformerPlayer(
            model=opponent_model,
            mode='sample',
            battle_format=self.config.BATTLE_FORMAT,
            server_configuration=self.config.SERVER_CONFIG,
            max_concurrent_battles=1,
            log_level=25,
        )

    async def _collect_rollout(self):
        """
        Play battles to fill the rollout buffer.
        Reuses persistent players to avoid challenge loop issues.
        """
        self.buffer.reset()

        # Update dense weight on reward shaper
        self.ppo_player.reward_shaper.dense_weight = self._get_dense_weight()

        # Refresh opponent every UPDATE_OPPONENT_EVERY updates
        if self.update_count % self.config.UPDATE_OPPONENT_EVERY == 0:
            self.opponent = self._make_opponent()
            await asyncio.sleep(1.0)

        battles_played = 0
        while not self.buffer.is_ready:
            await self.ppo_player.battle_against(
                self.opponent,
                n_battles=self.config.BATTLES_PER_ROLLOUT,
            )
            battles_played += self.config.BATTLES_PER_ROLLOUT
            await asyncio.sleep(0.3)

        self.total_steps += self.ppo_player.total_steps
        return self.ppo_player

    async def _evaluate(self) -> dict:
        """Evaluate current model."""
        results = {}
        await asyncio.sleep(1.0)  # let server settle

        eval_model = VGCPolicyNetwork().to(self.device)
        eval_model.load_state_dict(self.model.state_dict())
        eval_model.eval()

        # vs Random
        eval_player = TransformerPlayer(
            model=eval_model,
            mode='greedy',
            battle_format=self.config.BATTLE_FORMAT,
            server_configuration=self.config.SERVER_CONFIG,
            max_concurrent_battles=1,
            log_level=25,
        )
        random_opp = RandomPlayer(
            battle_format=self.config.BATTLE_FORMAT,
            server_configuration=self.config.SERVER_CONFIG,
            max_concurrent_battles=1,
            log_level=25,
        )
        n = self.config.EVAL_BATTLES
        await eval_player.battle_against(random_opp, n_battles=n)
        results['vs_random'] = eval_player.n_won_battles / n
        print(f"  vs Random:   {eval_player.n_won_battles}/{n} "
              f"({results['vs_random']:.1%})")

        await asyncio.sleep(1.0)

        # vs BC baseline
        eval_player2 = TransformerPlayer(
            model=eval_model,
            mode='greedy',
            battle_format=self.config.BATTLE_FORMAT,
            server_configuration=self.config.SERVER_CONFIG,
            max_concurrent_battles=1,
            log_level=25,
        )
        bc_model = VGCPolicyNetwork().to(self.device)
        bc_ckpt = torch.load(self.config.BC_CHECKPOINT, map_location=self.device)
        bc_model.load_state_dict(bc_ckpt['model_state_dict'])
        bc_opp = TransformerPlayer(
            model=bc_model,
            mode='greedy',
            battle_format=self.config.BATTLE_FORMAT,
            server_configuration=self.config.SERVER_CONFIG,
            max_concurrent_battles=1,
            log_level=25,
        )
        await eval_player2.battle_against(bc_opp, n_battles=n)
        results['vs_bc'] = eval_player2.n_won_battles / n
        print(f"  vs BC:       {eval_player2.n_won_battles}/{n} "
              f"({results['vs_bc']:.1%})")

        return results

    def _save_checkpoint(self, update: int, eval_results: Optional[dict] = None,
                         is_best: bool = False):
        checkpoint = {
            'update': update,
            'total_steps': self.total_steps,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.updater.optimizer.state_dict(),
            'eval_results': eval_results,
            'history': self.history,
        }
        path = os.path.join(
            self.config.CHECKPOINT_DIR, f'ppo_update_{update:04d}.pt'
        )
        torch.save(checkpoint, path)
        if is_best:
            best_path = os.path.join(self.config.CHECKPOINT_DIR, 'ppo_best.pt')
            torch.save(checkpoint, best_path)
            print(f"  *** New best model saved ***")

    async def train(self):
        """Main PPO training loop."""
        print(f"\n=== Starting PPO Training ===")
        print(f"Total updates:    {self.config.TOTAL_UPDATES}")
        print(f"Buffer size:      {self.config.BUFFER_SIZE}")
        print(f"Dense reward decay over {self.config.DENSE_REWARD_DECAY_STEPS} updates")
        print()

        start_update = self.update_count + 1
        end_update = self.update_count + self.config.TOTAL_UPDATES + 1
        for update in range(start_update, end_update):
            self.update_count = update
            update_start = time.time()

            # ── Collect rollout ───────────────────────────────────────────────
            ppo_player = await self._collect_rollout()

            # ── Compute GAE ───────────────────────────────────────────────────
            self.buffer.compute_gae(
                last_value=0.0,
                gamma=self.config.GAMMA,
                gae_lambda=self.config.GAE_LAMBDA,
            )

            # ── PPO update ────────────────────────────────────────────────────
            stats = self.updater.update(self.buffer)

            # ── Update opponent pool ──────────────────────────────────────────
            if update % self.config.UPDATE_OPPONENT_EVERY == 0:
                self.opponent_pool.add(self.model.state_dict())
                print(f"  Opponent pool updated. Size: {len(self.opponent_pool)}")

            update_time = time.time() - update_start

            # ── Logging ───────────────────────────────────────────────────────
            mean_reward = ppo_player.get_mean_episode_reward()
            dense_w = self._get_dense_weight()

            print(
                f"Update {update:4d}/{self.config.TOTAL_UPDATES} "
                f"| steps={self.total_steps:6d} "
                f"| loss={stats['total_loss']:.4f} "
                f"| policy={stats['policy_loss']:.4f} "
                f"| value={stats['value_loss']:.4f} "
                f"| entropy={stats['entropy']:.4f} "
                f"| kl={stats['approx_kl']:.4f} "
                f"| reward={mean_reward:.3f} "
                f"| dense_w={dense_w:.2f} "
                f"| {update_time:.0f}s"
            )

            # ── Evaluation ────────────────────────────────────────────────────
            eval_results = None
            if update % self.config.EVAL_EVERY == 0:
                print(f"\nEvaluating at update {update}...")
                eval_results = await self._evaluate()

                vs_random = eval_results.get('vs_random', 0)
                is_best = vs_random > self.best_win_rate
                if is_best:
                    self.best_win_rate = vs_random

                self._save_checkpoint(update, eval_results, is_best)

            elif update % self.config.SAVE_EVERY == 0:
                self._save_checkpoint(update)

            # ── History ───────────────────────────────────────────────────────
            self.history.append({
                'update': update,
                'total_steps': self.total_steps,
                'stats': stats,
                'mean_reward': mean_reward,
                'dense_weight': dense_w,
                'eval': eval_results,
            })

        print(f"\n=== PPO Training Complete ===")
        print(f"Total steps:    {self.total_steps:,}")
        print(f"Best win rate:  {self.best_win_rate:.1%} vs Random")

        # Save final checkpoint
        self._save_checkpoint(self.config.TOTAL_UPDATES)
        return self.history


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    config = PPOConfig()
    trainer = PPOTrainer(config)
    asyncio.run(trainer.train())