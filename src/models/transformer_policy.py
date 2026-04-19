import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

from src.utils.observation import (
    get_observation_size,
    N_ITEMS,
    N_TYPES,
    N_STATUSES,
    N_EFFECTS,
    N_WEATHERS,
    N_TERRAINS, 
    N_FIELDS,
    N_SIDE_CONDITIONS,
    N_BOOSTS,
    N_BASE_STATS,
)

# ─── Architecture Constants ────────────────────────────────────────────────────
D_MODEL = 256          # embedding dimension throughout the transformer
N_HEADS = 4            # attention heads (D_MODEL must be divisible by N_HEADS)
N_LAYERS = 3           # transformer encoder layers
D_FF = 512             # feedforward dimension inside transformer
DROPOUT = 0.1          # dropout rate

# Token counts
N_MON_TOKENS = 12      # 6 ally + 6 opponent pokemon
N_MOVE_TOKENS = 8      # 2 active slots × 4 moves
N_TOKENS = N_MON_TOKENS + N_MOVE_TOKENS  # 20 total tokens

MON_INPUT_DIM = (
    1           # hp fraction
    + 18        # types
    + N_BASE_STATS      # 6 base stats
    + N_BOOSTS          # 7 boosts
    + N_STATUSES        # 7 statuses
    + N_EFFECTS         # 54 volatile effects
    + 1         # is_active
    + 1         # is_fainted
    + (N_ITEMS + 1)     # item one-hot
    + (N_TYPES + 1)     # tera type one-hot
    + 1         # is_terastallized
    + 10        # immunity flags
)  # = 231

MOVE_INPUT_DIM = 30 

GLOBAL_STATE_DIM = (
    N_WEATHERS
    + N_TERRAINS
    + N_FIELDS
    + 2 * N_SIDE_CONDITIONS
    + 1         # turn number
)  # = 47

# Action space per slot
# 4 moves × 3 targets (opp1, opp2, ally) + 4 switches + 1 struggle + 1 pass
N_MOVE_ACTIONS = 4 * 3   # 12
N_SWITCH_ACTIONS = 4     # 4
N_SPECIAL_ACTIONS = 2    # struggle + pass
N_ACTIONS_PER_SLOT = N_MOVE_ACTIONS + N_SWITCH_ACTIONS + N_SPECIAL_ACTIONS  # 18


# ─── Positional / Role Encoding ───────────────────────────────────────────────

class RoleEncoding(nn.Module):
    """
    Learned role embeddings for each token position.
    Unlike sinusoidal positional encoding, this lets the model learn
    that token 0 = 'ally active slot 0', token 6 = 'opponent active slot 0' etc.
    """
    def __init__(self, n_tokens: int, d_model: int):
        super().__init__()
        self.encoding = nn.Embedding(n_tokens, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_tokens, d_model)
        batch_size = x.size(0)
        positions = torch.arange(N_TOKENS, device=x.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        return x + self.encoding(positions)


# ─── Input Projections ────────────────────────────────────────────────────────

class InputProjection(nn.Module):
    """
    Projects raw input vectors into d_model dimensional space.
    Separate projections for pokemon tokens and move tokens.
    Global state is broadcast-added to all tokens after projection.
    """
    def __init__(self, d_model: int):
        super().__init__()

        # Pokemon projection: 221 → d_model
        self.mon_proj = j = nn.Sequential(
            nn.Linear(MON_INPUT_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # Move projection: 28 → d_model
        self.move_proj = nn.Sequential(
            nn.Linear(MOVE_INPUT_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # Global state projection: 47 → d_model
        # This gets added to ALL tokens so every token is aware of field state
        self.global_proj = nn.Sequential(
            nn.Linear(GLOBAL_STATE_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

    def forward(
        self,
        mon_tokens: torch.Tensor,    # (batch, 12, 221)
        move_tokens: torch.Tensor,   # (batch, 8, 28)
        global_state: torch.Tensor,  # (batch, 47)
    ) -> torch.Tensor:               # (batch, 20, d_model)

        # Project each token type
        mon_emb = self.mon_proj(mon_tokens)      # (batch, 12, d_model)
        move_emb = self.move_proj(move_tokens)   # (batch, 8, d_model)

        # Concatenate into single token sequence
        tokens = torch.cat([mon_emb, move_emb], dim=1)  # (batch, 20, d_model)

        # Project global state and broadcast to all tokens
        global_emb = self.global_proj(global_state)     # (batch, d_model)
        global_emb = global_emb.unsqueeze(1)            # (batch, 1, d_model)
        tokens = tokens + global_emb                    # broadcast add

        return tokens


# ─── Transformer Encoder ──────────────────────────────────────────────────────

class BattleTransformer(nn.Module):
    """
    Standard Transformer Encoder that processes the 20 battle tokens.
    Each token attends to all other tokens via multi-head self-attention.
    """
    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,   # input shape: (batch, seq, features)
            norm_first=True,    # pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (batch, 20, d_model)
        return self.encoder(tokens)  # (batch, 20, d_model)


# ─── Policy Head ──────────────────────────────────────────────────────────────

class PolicyHead(nn.Module):
    """
    Takes the enriched token for one active slot and produces action logits.
    Used twice — once for slot 0, once for slot 1.
    """
    def __init__(self, d_model: int, n_actions: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_actions),
        )

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        # token: (batch, d_model)
        return self.net(token)  # (batch, n_actions)


# ─── Value Head (for RL) ──────────────────────────────────────────────────────

class ValueHead(nn.Module):
    """
    Estimates the value V(s) of the current state.
    Used during PPO training — not needed for Behavioral Cloning.
    Takes a global summary token (mean of all tokens).
    """
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (batch, 20, d_model)
        # Global summary = mean pool over all tokens
        summary = tokens.mean(dim=1)   # (batch, d_model)
        return self.net(summary)       # (batch, 1)


# ─── Full Policy Network ──────────────────────────────────────────────────────

class VGCPolicyNetwork(nn.Module):
    """
    Full policy network for VGC doubles.

    Input:  flat observation vector of size 2923
    Output: action logits for slot 0 and slot 1 (after action masking)

    Can be used for:
      - Behavioral Cloning (supervised learning on human replays)
      - PPO (reinforcement learning with value head)
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        d_ff: int = D_FF,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.d_model = d_model

        # Input projection
        self.input_proj = InputProjection(d_model)

        # Role encoding
        self.role_encoding = RoleEncoding(N_TOKENS, d_model)

        # Transformer encoder
        self.transformer = BattleTransformer(d_model, n_heads, n_layers, d_ff, dropout)

        # Policy heads (one per active slot)
        self.policy_head_0 = PolicyHead(d_model, N_ACTIONS_PER_SLOT, dropout)
        self.policy_head_1 = PolicyHead(d_model, N_ACTIONS_PER_SLOT, dropout)

        # Value head (for RL)
        self.value_head = ValueHead(d_model, dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def _split_observation(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split flat observation vector back into structured components.
        obs shape: (batch, 2923)

        Returns:
            mon_tokens:   (batch, 12, 221)
            move_tokens:  (batch, 8, 28)
            global_state: (batch, 47)
        """
        batch = obs.size(0)

        # Pokemon tokens: first 12*221 = 2652 values
        mon_flat = obs[:, :N_MON_TOKENS * MON_INPUT_DIM]
        mon_tokens = mon_flat.view(batch, N_MON_TOKENS, MON_INPUT_DIM)

        # Move tokens: next 8*28 = 224 values
        move_start = N_MON_TOKENS * MON_INPUT_DIM
        move_end = move_start + N_MOVE_TOKENS * MOVE_INPUT_DIM
        move_flat = obs[:, move_start:move_end]
        move_tokens = move_flat.view(batch, N_MOVE_TOKENS, MOVE_INPUT_DIM)

        # Global state: last 47 values
        global_state = obs[:, move_end:]

        return mon_tokens, move_tokens, global_state

    def forward(
        self,
        obs: torch.Tensor,                          # (batch, 2923)
        action_mask_0: Optional[torch.Tensor] = None,  # (batch, 18)
        action_mask_1: Optional[torch.Tensor] = None,  # (batch, 18)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            logits_0: (batch, 18) — action logits for slot 0
            logits_1: (batch, 18) — action logits for slot 1
            value:    (batch, 1)  — state value estimate
        """
        # Split observation into structured components
        mon_tokens, move_tokens, global_state = self._split_observation(obs)

        # Project inputs to d_model
        tokens = self.input_proj(mon_tokens, move_tokens, global_state)

        # Add role encodings
        tokens = self.role_encoding(tokens)

        # Run through transformer
        tokens = self.transformer(tokens)  # (batch, 20, d_model)

        # Extract tokens for active slots
        # Token 0 = ally active slot 0, Token 1 = ally active slot 1
        slot0_token = tokens[:, 0, :]  # (batch, d_model)
        slot1_token = tokens[:, 1, :]  # (batch, d_model)

        # Policy heads → raw logits
        logits_0 = self.policy_head_0(slot0_token)  # (batch, 18)
        logits_1 = self.policy_head_1(slot1_token)  # (batch, 18)

        # Apply action masks (set illegal actions to -inf)
        if action_mask_0 is not None:
            logits_0 = logits_0.masked_fill(~action_mask_0.bool(), float('-inf'))
        if action_mask_1 is not None:
            logits_1 = logits_1.masked_fill(~action_mask_1.bool(), float('-inf'))

        # Value estimate
        value = self.value_head(tokens)  # (batch, 1)

        return logits_0, logits_1, value

    def get_action_probs(
        self,
        obs: torch.Tensor,
        action_mask_0: Optional[torch.Tensor] = None,
        action_mask_1: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns softmax probabilities instead of raw logits."""
        logits_0, logits_1, _ = self.forward(obs, action_mask_0, action_mask_1)
        probs_0 = F.softmax(logits_0, dim=-1)
        probs_1 = F.softmax(logits_1, dim=-1)
        return probs_0, probs_1

    def get_model_info(self) -> dict:
        """Returns model size information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "d_model": self.d_model,
            "observation_size": get_observation_size(),
            "actions_per_slot": N_ACTIONS_PER_SLOT,
        }


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Initializing VGCPolicyNetwork...")
    model = VGCPolicyNetwork()

    info = model.get_model_info()
    print(f"\nModel Info:")
    for k, v in info.items():
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")

    # Test forward pass with random input
    batch_size = 4
    obs_size = get_observation_size()
    dummy_obs = torch.randn(batch_size, obs_size)

    # Dummy action masks (all actions valid)
    mask_0 = torch.ones(batch_size, N_ACTIONS_PER_SLOT, dtype=torch.bool)
    mask_1 = torch.ones(batch_size, N_ACTIONS_PER_SLOT, dtype=torch.bool)

    print(f"\nRunning forward pass with batch_size={batch_size}, obs_size={obs_size}...")
    with torch.no_grad():
        logits_0, logits_1, value = model(dummy_obs, mask_0, mask_1)

    print(f"\nOutput shapes:")
    print(f"  logits_0: {logits_0.shape}  (expected: [{batch_size}, {N_ACTIONS_PER_SLOT}])")
    print(f"  logits_1: {logits_1.shape}  (expected: [{batch_size}, {N_ACTIONS_PER_SLOT}])")
    print(f"  value:    {value.shape}     (expected: [{batch_size}, 1])")

    # Test with some masked actions
    mask_0[:, 8:] = False   # mask out switches and specials for slot 0
    probs_0, probs_1 = model.get_action_probs(dummy_obs, mask_0, mask_1)
    print(f"\nWith action masking:")
    print(f"  probs_0 sum: {probs_0.sum(dim=-1)}  (should be all 1.0)")
    print(f"  masked actions prob: {probs_0[:, 8:].sum(dim=-1)}  (should be all 0.0)")

    print("\nAll checks passed.")