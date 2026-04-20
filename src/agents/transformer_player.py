import torch
import numpy as np
from typing import Optional
from poke_env.player import Player
from poke_env.battle.double_battle import DoubleBattle
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder, DefaultBattleOrder

from src.models.transformer_policy import VGCPolicyNetwork
from src.utils.observation import embed_battle
from src.utils.action_space import (
    get_action_masks_tensor,
    actions_to_double_order,
    get_action_mask,
    PASS_ACTION,
)


class TransformerPlayer(Player):
    """
    A poke-env Player that uses the VGCPolicyNetwork to make decisions.

    Supports three modes:
        'greedy'  — always picks the highest probability action (for evaluation)
        'sample'  — samples from the probability distribution (for training/exploration)
        'random'  — ignores the network, plays randomly (for debugging)
    """

    def __init__(
        self,
        model: Optional[VGCPolicyNetwork] = None,
        device: Optional[torch.device] = None,
        mode: str = "greedy",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Device setup
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # Model setup
        self.model = model or VGCPolicyNetwork()
        self.model.to(self.device)
        self.model.eval()

        assert mode in {"greedy", "sample", "random"}, \
            f"mode must be 'greedy', 'sample', or 'random', got '{mode}'"
        self.mode = mode

        # Battle tracking for analysis
        self.turn_data = []   # stores (obs, action_0, action_1, mask_0, mask_1) per turn

    def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        # Random mode — bypass network entirely
        if self.mode == "random":
            return self.choose_random_doubles_move(battle)

       # Handle full force switch
        if any(battle.force_switch):
            return self.choose_random_doubles_move(battle)

        try:
            # ── Build observation vector ──
            obs = embed_battle(battle)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            # obs_tensor shape: (1, 2923)

            # ── Build action masks ───
            mask_0, mask_1 = get_action_masks_tensor(battle, device=self.device)
            # mask shapes: (1, 18)

            # ── Forward pass ──
            with torch.no_grad():
                logits_0, logits_1, value = self.model(obs_tensor, mask_0, mask_1)
            # logits shapes: (1, 18)

            # ── Select actions ──
            action_0 = self._select_action(logits_0, mask_0)
            action_1 = self._select_action(logits_1, mask_1)

            # ── Store turn data for later analysis/training ──
            self.turn_data.append({
                "turn": battle.turn,
                "obs": obs,
                "action_0": action_0,
                "action_1": action_1,
                "mask_0": mask_0.cpu().numpy(),
                "mask_1": mask_1.cpu().numpy(),
                "value": value.item(),
            })

            # ── Convert actions to battle order ──
            return actions_to_double_order(action_0, action_1, battle)

        except Exception as e:
            # Fallback to random on any error — log it
            self.logger.error(f"TransformerPlayer error on turn {battle.turn}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self.choose_random_doubles_move(battle)

    def _select_action(
        self,
        logits: torch.Tensor,   # (1, 18)
        mask: torch.Tensor,     # (1, 18) bool
    ) -> int:
        """Select an action from logits given a validity mask."""

        # Safety check — if all actions masked, return pass
        if not mask.any():
            return PASS_ACTION

        if self.mode == "greedy":
            # Pick highest logit valid action
            action = logits.argmax(dim=-1).item()

        elif self.mode == "sample":
            # Sample from softmax distribution
            import torch.nn.functional as F
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1).item()

        return action

    def reset_turn_data(self):
        """Clear stored turn data between battles."""
        self.turn_data = []

    def set_mode(self, mode: str):
        """Switch between greedy/sample/random modes."""
        assert mode in {"greedy", "sample", "random"}
        self.mode = mode

    def set_model(self, model: VGCPolicyNetwork):
        """Swap in a new model (e.g. after training)."""
        self.model = model
        self.model.to(self.device)
        self.model.eval()