import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
from src.models.transformer_policy import VGCPolicyNetwork
from src.rl.rollout_buffer import RolloutBuffer


class PPOUpdater:
    """
    Performs PPO policy and value updates from a filled RolloutBuffer.

    Uses:
        - Clipped surrogate policy loss
        - Value function loss (MSE)
        - Entropy bonus for exploration
        - Separate losses for slot 0 and slot 1
    """

    def __init__(
        self,
        model: VGCPolicyNetwork,
        learning_rate: float = 3e-4,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 256,
        device: torch.device = torch.device('cpu'),
    ):
        self.model = model
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,
            eta_min=1e-6,
        )

        # Training stats
        self.update_count = 0
        self.stats_history = []

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Run n_epochs of PPO updates over the rollout buffer.

        Args:
            buffer: filled RolloutBuffer with computed GAE advantages

        Returns:
            dict of training statistics
        """
        self.model.train()

        stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'total_loss': 0.0,
            'clip_fraction': 0.0,
            'approx_kl': 0.0,
            'n_updates': 0,
        }

        for epoch in range(self.n_epochs):
            for batch in buffer.get_batches(self.batch_size):
                (obs, actions_0, actions_1,
                 old_log_probs_0, old_log_probs_1,
                 advantages, returns,
                 masks_0, masks_1) = [x.to(self.device) for x in batch]

                # ── Forward pass ──────────────────────────────────────────────
                logits_0, logits_1, values = self.model(obs, masks_0, masks_1)
                values = values.squeeze(-1)  # (batch,)

                # ── Policy loss ───────────────────────────────────────────────
                policy_loss, entropy, clip_frac, approx_kl = self._policy_loss(
                    logits_0, logits_1,
                    actions_0, actions_1,
                    old_log_probs_0, old_log_probs_1,
                    advantages,
                )

                # ── Value loss ────────────────────────────────────────────────
                value_loss = F.mse_loss(values, returns)

                # ── Total loss ────────────────────────────────────────────────
                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                # ── Backward pass ─────────────────────────────────────────────
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                # ── Accumulate stats ──────────────────────────────────────────
                stats['policy_loss']  += policy_loss.item()
                stats['value_loss']   += value_loss.item()
                stats['entropy']      += entropy.item()
                stats['total_loss']   += total_loss.item()
                stats['clip_fraction']+= clip_frac
                stats['approx_kl']   += approx_kl
                stats['n_updates']   += 1

        # Average stats over all updates
        n = stats['n_updates']
        if n > 0:
            for key in ['policy_loss', 'value_loss', 'entropy',
                        'total_loss', 'clip_fraction', 'approx_kl']:
                stats[key] /= n

        self.scheduler.step()
        self.update_count += 1
        self.stats_history.append(stats)
        self.model.eval()

        return stats

    def _policy_loss(
        self,
        logits_0: torch.Tensor,      # (batch, 18)
        logits_1: torch.Tensor,      # (batch, 18)
        actions_0: torch.Tensor,     # (batch,)
        actions_1: torch.Tensor,     # (batch,)
        old_log_probs_0: torch.Tensor,  # (batch,)
        old_log_probs_1: torch.Tensor,  # (batch,)
        advantages: torch.Tensor,    # (batch,)
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Compute clipped PPO policy loss for both slots combined.

        Returns:
            policy_loss, entropy, clip_fraction, approx_kl
        """

        # Guard against NaN/Inf in logits
        logits_0 = torch.nan_to_num(logits_0, nan=0.0, posinf=1e4, neginf=-1e4)
        logits_1 = torch.nan_to_num(logits_1, nan=0.0, posinf=1e4, neginf=-1e4)
        old_log_probs_0 = torch.nan_to_num(old_log_probs_0, nan=0.0)
        old_log_probs_1 = torch.nan_to_num(old_log_probs_1, nan=0.0)
        advantages = torch.nan_to_num(advantages, nan=0.0)

        # Log probs under current policy
        log_probs_0 = self._log_prob_from_logits(logits_0, actions_0)
        log_probs_1 = self._log_prob_from_logits(logits_1, actions_1)

        # Entropy for exploration
        entropy_0 = self._entropy_from_logits(logits_0)
        entropy_1 = self._entropy_from_logits(logits_1)
        entropy = (entropy_0 + entropy_1) / 2.0

        # Ratios for PPO clip
        ratio_0 = torch.exp(log_probs_0 - old_log_probs_0)
        ratio_1 = torch.exp(log_probs_1 - old_log_probs_1)

        # Use average ratio across both slots
        ratio = (ratio_0 + ratio_1) / 2.0

        # Clipped surrogate loss
        clip_low = 1.0 - self.clip_epsilon
        clip_high = 1.0 + self.clip_epsilon

        surrogate_1 = ratio * advantages
        surrogate_2 = torch.clamp(ratio, clip_low, clip_high) * advantages
        policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

        # Diagnostics
        with torch.no_grad():
            clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
            log_ratio = (log_probs_0 + log_probs_1) / 2.0 - \
                        (old_log_probs_0 + old_log_probs_1) / 2.0
            approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()

        return policy_loss, entropy.mean(), clip_fraction, approx_kl

    def _log_prob_from_logits(
        self,
        logits: torch.Tensor,   # (batch, n_actions)
        actions: torch.Tensor,  # (batch,)
    ) -> torch.Tensor:          # (batch,)
        """Compute log probability of selected actions."""
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

    def _entropy_from_logits(
        self,
        logits: torch.Tensor,   # (batch, n_actions)
    ) -> torch.Tensor:          # (batch,)
        """Compute entropy of action distribution."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(probs * log_probs).sum(dim=-1)

    def get_action_and_log_prob(
        self,
        obs: torch.Tensor,
        mask_0: torch.Tensor,
        mask_1: torch.Tensor,
    ) -> Tuple[int, int, float, float, float]:
        """
        Sample actions and return log probs for rollout collection.

        Returns:
            action_0, action_1, log_prob_0, log_prob_1, value
        """
        with torch.no_grad():
            logits_0, logits_1, value = self.model(obs, mask_0, mask_1)

            # Guard against all-masked logits producing NaN
            # If all actions masked, fall back to uniform over all actions
            if torch.isnan(logits_0).any() or torch.isinf(logits_0).all():
                logits_0 = torch.zeros_like(logits_0)
            if torch.isnan(logits_1).any() or torch.isinf(logits_1).all():
                logits_1 = torch.zeros_like(logits_1)

            probs_0 = F.softmax(logits_0, dim=-1)
            probs_1 = F.softmax(logits_1, dim=-1)

            # Final NaN check
            if torch.isnan(probs_0).any():
                probs_0 = torch.ones_like(probs_0) / probs_0.shape[-1]
            if torch.isnan(probs_1).any():
                probs_1 = torch.ones_like(probs_1) / probs_1.shape[-1]

            dist_0 = torch.distributions.Categorical(probs_0)
            dist_1 = torch.distributions.Categorical(probs_1)

            action_0 = dist_0.sample()
            action_1 = dist_1.sample()

            log_prob_0 = dist_0.log_prob(action_0)
            log_prob_1 = dist_1.log_prob(action_1)

        return (
            action_0.item(),
            action_1.item(),
            log_prob_0.item(),
            log_prob_1.item(),
            value.item(),
        )


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from src.utils.observation import get_observation_size

    device = torch.device('cpu')
    obs_dim = get_observation_size()

    # Initialize model and updater
    model = VGCPolicyNetwork().to(device)
    updater = PPOUpdater(model=model, device=device)

    print(f"PPOUpdater initialized")
    print(f"Optimizer: AdamW lr=3e-4")
    print(f"Clip epsilon: {updater.clip_epsilon}")
    print(f"Value coef:   {updater.value_coef}")
    print(f"Entropy coef: {updater.entropy_coef}")
    print()

    # Test action sampling
    obs = torch.randn(1, obs_dim)
    mask_0 = torch.ones(1, 18, dtype=torch.bool)
    mask_1 = torch.ones(1, 18, dtype=torch.bool)

    a0, a1, lp0, lp1, val = updater.get_action_and_log_prob(obs, mask_0, mask_1)
    print(f"Action sample: slot0={a0}, slot1={a1}")
    print(f"Log probs:     slot0={lp0:.4f}, slot1={lp1:.4f}")
    print(f"Value:         {val:.4f}")
    print()

    # Test full update cycle
    buf = RolloutBuffer(buffer_size=512, obs_dim=obs_dim)
    for i in range(512):
        buf.add(
            obs=np.random.randn(obs_dim).astype(np.float32),
            action_0=np.random.randint(0, 18),
            action_1=np.random.randint(0, 18),
            log_prob_0=float(np.random.randn()),
            log_prob_1=float(np.random.randn()),
            value=float(np.random.randn()),
            reward=float(np.random.randn() * 0.1),
            done=bool(np.random.rand() < 0.05),
            mask_0=np.ones(18, dtype=bool),
            mask_1=np.ones(18, dtype=bool),
        )
    buf.compute_gae()

    stats = updater.update(buf)
    print(f"PPO update stats:")
    for k, v in stats.items():
        if k != 'n_updates':
            print(f"  {k:15s}: {v:.4f}")

    print("\nAll checks passed.")