import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from poke_env.player import Player
from poke_env.battle.double_battle import DoubleBattle
from poke_env.player.battle_order import BattleOrder

from src.models.transformer_policy import VGCPolicyNetwork
from src.utils.observation import embed_battle
from src.utils.action_space import (
    get_action_masks_tensor,
    actions_to_double_order,
    PASS_ACTION,
)
from src.rl.rollout_buffer import RolloutBuffer
from src.rl.reward_shaper import RewardShaper, BattleState
from src.rl.ppo_updater import PPOUpdater


class PPOPlayer(Player):
    """
    A poke-env Player that collects PPO rollouts during battles.

    During each turn:
    1. Embeds the battle state into an observation vector
    2. Samples actions from the policy
    3. Records (obs, action, log_prob, value, mask) in the rollout buffer
    4. On battle end, computes the terminal reward and marks done=True
    """

    def __init__(
        self,
        model: VGCPolicyNetwork,
        buffer: RolloutBuffer,
        updater: PPOUpdater,
        reward_shaper: RewardShaper,
        device: torch.device,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.buffer = buffer
        self.updater = updater
        self.reward_shaper = reward_shaper
        self.device = device

        # Per-battle tracking
        self._prev_battle_state: Optional[BattleState] = None
        self._last_obs: Optional[np.ndarray] = None
        self._last_action_0: Optional[int] = None
        self._last_action_1: Optional[int] = None
        self._last_log_prob_0: Optional[float] = None
        self._last_log_prob_1: Optional[float] = None
        self._last_value: Optional[float] = None
        self._last_mask_0: Optional[np.ndarray] = None
        self._last_mask_1: Optional[np.ndarray] = None

        # Stats
        self.total_steps = 0
        self.episode_rewards = []
        self._current_episode_reward = 0.0

    def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        """Called every turn — collect rollout data and return action."""

        # Handle force switch
        if any(battle.force_switch):
            # Still need to store transition from previous turn
            self._store_previous_transition(battle, done=False)
            return self.choose_random_doubles_move(battle)

        try:
            # ── Store previous transition ──────────────────────────────────
            # We store the previous turn's data now that we know its reward
            self._store_previous_transition(battle, done=False)

            # ── Build current observation ──────────────────────────────────
            obs = embed_battle(battle)
            obs_tensor = torch.tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            # ── Build action masks ─────────────────────────────────────────
            mask_0_tensor, mask_1_tensor = get_action_masks_tensor(
                battle, device=self.device
            )

            # ── Sample actions from policy ─────────────────────────────────
            action_0, action_1, log_prob_0, log_prob_1, value = \
                self.updater.get_action_and_log_prob(
                    obs_tensor, mask_0_tensor, mask_1_tensor
                )

            # ── Store current state for next turn ──────────────────────────
            self._last_obs = obs
            self._last_action_0 = action_0
            self._last_action_1 = action_1
            self._last_log_prob_0 = log_prob_0
            self._last_log_prob_1 = log_prob_1
            self._last_value = value
            self._last_mask_0 = mask_0_tensor.cpu().numpy().squeeze(0)
            self._last_mask_1 = mask_1_tensor.cpu().numpy().squeeze(0)

            # Update reward shaper state
            self._prev_battle_state = BattleState(battle)

            # ── Return battle order ────────────────────────────────────────
            return actions_to_double_order(action_0, action_1, battle)

        except Exception as e:
            self.logger.error(f"PPOPlayer error on turn {battle.turn}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self._store_previous_transition(battle, done=False)
            return self.choose_random_doubles_move(battle)

    def _store_previous_transition(
        self,
        battle: DoubleBattle,
        done: bool,
    ):
        """Store the previous turn's transition in the buffer."""
        if self._last_obs is None:
            # First turn — initialize reward shaper
            self.reward_shaper.reset()
            self._prev_battle_state = BattleState(battle)
            return

        # Compute reward for previous action
        reward = self.reward_shaper.compute_reward(
            battle,
            prev_state=self._prev_battle_state,
        )
        self._current_episode_reward += reward

        # Store in buffer
        self.buffer.add(
            obs=self._last_obs,
            action_0=self._last_action_0,
            action_1=self._last_action_1,
            log_prob_0=self._last_log_prob_0,
            log_prob_1=self._last_log_prob_1,
            value=self._last_value,
            reward=reward,
            done=done,
            mask_0=self._last_mask_0,
            mask_1=self._last_mask_1,
        )
        self.total_steps += 1

    def _battle_finished_callback(self, battle: DoubleBattle):
        """Called by poke-env when a battle ends."""
        # Store final transition with terminal reward
        self._store_previous_transition(battle, done=True)

        # Record episode reward
        self.episode_rewards.append(self._current_episode_reward)
        self._current_episode_reward = 0.0

        # Reset per-battle state
        self._last_obs = None
        self._last_action_0 = None
        self._last_action_1 = None
        self._last_log_prob_0 = None
        self._last_log_prob_1 = None
        self._last_value = None
        self._last_mask_0 = None
        self._last_mask_1 = None
        self._prev_battle_state = None
        self.reward_shaper.reset()

    def get_mean_episode_reward(self, last_n: int = 10) -> float:
        """Returns mean reward over last n episodes."""
        if not self.episode_rewards:
            return 0.0
        rewards = self.episode_rewards[-last_n:]
        return sum(rewards) / len(rewards)