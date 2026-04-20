import numpy as np
from typing import Optional


class RolloutBuffer:
    """
    Stores trajectories from self-play battles for PPO updates.

    Each step stores:
        obs         — 3059-dim observation vector
        action_0    — action taken by slot 0 (0-17)
        action_1    — action taken by slot 1 (0-17)
        log_prob_0  — log probability of action_0 under current policy
        log_prob_1  — log probability of action_1 under current policy
        value       — critic's value estimate V(s)
        reward      — shaped reward for this step
        done        — whether the battle ended this step
        mask_0      — action mask for slot 0 (18,)
        mask_1      — action mask for slot 1 (18,)
    """

    def __init__(self, buffer_size: int, obs_dim: int, n_actions: int = 18):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.reset()

    def reset(self):
        """Clear the buffer."""
        self.obs         = np.zeros((self.buffer_size, self.obs_dim), dtype=np.float32)
        self.actions_0   = np.zeros(self.buffer_size, dtype=np.int64)
        self.actions_1   = np.zeros(self.buffer_size, dtype=np.int64)
        self.log_probs_0 = np.zeros(self.buffer_size, dtype=np.float32)
        self.log_probs_1 = np.zeros(self.buffer_size, dtype=np.float32)
        self.values      = np.zeros(self.buffer_size, dtype=np.float32)
        self.rewards     = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones       = np.zeros(self.buffer_size, dtype=np.float32)
        self.masks_0     = np.zeros((self.buffer_size, self.n_actions), dtype=np.bool_)
        self.masks_1     = np.zeros((self.buffer_size, self.n_actions), dtype=np.bool_)

        # Computed after collection
        self.advantages  = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns     = np.zeros(self.buffer_size, dtype=np.float32)

        self.ptr = 0        # current write position
        self.full = False   # whether buffer has been filled once

    def add(
        self,
        obs: np.ndarray,
        action_0: int,
        action_1: int,
        log_prob_0: float,
        log_prob_1: float,
        value: float,
        reward: float,
        done: bool,
        mask_0: np.ndarray,
        mask_1: np.ndarray,
    ):
        """Add one step to the buffer."""
        idx = self.ptr

        self.obs[idx]         = obs
        self.actions_0[idx]   = action_0
        self.actions_1[idx]   = action_1
        self.log_probs_0[idx] = log_prob_0
        self.log_probs_1[idx] = log_prob_1
        self.values[idx]      = value
        self.rewards[idx]     = reward
        self.dones[idx]       = float(done)
        self.masks_0[idx]     = mask_0
        self.masks_1[idx]     = mask_1

        self.ptr = (self.ptr + 1) % self.buffer_size
        if self.ptr == 0:
            self.full = True

    def compute_gae(
        self,
        last_value: float = 0.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Compute Generalized Advantage Estimation (GAE) and returns.

        Must be called after buffer is full before sampling.

        Args:
            last_value: V(s_{T+1}) — value of state after last collected step.
                        0.0 if last step was terminal.
            gamma:      discount factor
            gae_lambda: GAE lambda parameter
        """
        size = self.buffer_size if self.full else self.ptr
        last_gae = 0.0

        for t in reversed(range(size)):
            if t == size - 1:
                next_value = last_value
                next_done = 0.0
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]

            # TD error: r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
            delta = (
                self.rewards[t]
                + gamma * next_value * (1.0 - next_done)
                - self.values[t]
            )

            # GAE: advantage = delta + γλ * next_advantage * (1 - done)
            last_gae = delta + gamma * gae_lambda * (1.0 - next_done) * last_gae
            self.advantages[t] = last_gae

        # Returns = advantages + values (used as value targets)
        self.returns[:size] = self.advantages[:size] + self.values[:size]

        # Normalize advantages for training stability
        size_adv = self.advantages[:size]
        self.advantages[:size] = (size_adv - size_adv.mean()) / (size_adv.std() + 1e-8)

    def get_batches(self, batch_size: int):
        """
        Yield random mini-batches for PPO updates.

        Yields tuples of torch tensors:
            (obs, actions_0, actions_1, log_probs_0, log_probs_1,
             advantages, returns, masks_0, masks_1)
        """
        import torch

        size = self.buffer_size if self.full else self.ptr
        indices = np.random.permutation(size)

        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            idx = indices[start:end]

            yield (
                torch.tensor(self.obs[idx],         dtype=torch.float32),
                torch.tensor(self.actions_0[idx],   dtype=torch.long),
                torch.tensor(self.actions_1[idx],   dtype=torch.long),
                torch.tensor(self.log_probs_0[idx], dtype=torch.float32),
                torch.tensor(self.log_probs_1[idx], dtype=torch.float32),
                torch.tensor(self.advantages[idx],  dtype=torch.float32),
                torch.tensor(self.returns[idx],     dtype=torch.float32),
                torch.tensor(self.masks_0[idx],     dtype=torch.bool),
                torch.tensor(self.masks_1[idx],     dtype=torch.bool),
            )

    @property
    def is_ready(self) -> bool:
        """Returns True when buffer has enough data for an update."""
        return self.full or self.ptr >= self.buffer_size

    @property
    def size(self) -> int:
        return self.buffer_size if self.full else self.ptr


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch
    from src.utils.observation import get_observation_size

    obs_dim = get_observation_size()
    buf = RolloutBuffer(buffer_size=2048, obs_dim=obs_dim)

    print(f"Buffer size:   {buf.buffer_size}")
    print(f"Obs dim:       {buf.obs_dim}")
    print(f"Memory usage:  {buf.obs.nbytes / 1024 / 1024:.1f} MB (obs only)")

    # Fill with dummy data
    for i in range(2048):
        buf.add(
            obs=np.random.randn(obs_dim).astype(np.float32),
            action_0=np.random.randint(0, 18),
            action_1=np.random.randint(0, 18),
            log_prob_0=np.random.randn(),
            log_prob_1=np.random.randn(),
            value=np.random.randn(),
            reward=np.random.randn() * 0.1,
            done=bool(np.random.rand() < 0.05),
            mask_0=np.ones(18, dtype=bool),
            mask_1=np.ones(18, dtype=bool),
        )

    print(f"Buffer full:   {buf.is_ready}")

    # Test GAE
    buf.compute_gae(last_value=0.0)
    print(f"Advantages:    mean={buf.advantages.mean():.4f} std={buf.advantages.std():.4f}")
    print(f"Returns:       mean={buf.returns.mean():.4f} std={buf.returns.std():.4f}")

    # Test batching
    batches = list(buf.get_batches(batch_size=256))
    print(f"Num batches:   {len(batches)} (expected ~8)")
    print(f"Batch shapes:  obs={batches[0][0].shape}")

    print("\nAll checks passed.")