import numpy as np
import torch
from typing import List, Optional, Tuple
from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.player.battle_order import (
    DoubleBattleOrder,
    SingleBattleOrder,
    PassBattleOrder,
    DefaultBattleOrder,
)

# ─── Action Space Layout ───────────────────────────────────────────────────────
#
# Each slot has 18 possible actions:
#
# Index  0: Move 0 → tart_t opponent slot 1  (OPPONENT_1_POSITION)
# Index  1: Move 0 → target opponent slot 2  (OPPONENT_2_POSITION)
# Index  2: Move 0 → target ally             (ALLY position, for moves like Helping Hand)
# Index  3: Move 1 → target opponent slot 1
# Index  4: Move 1 → target opponent slot 2
# Index  5: Move 1 → target ally
# Index  6: Move 2 → target opponent slot 1
# Index  7: Move 2 → target opponent slot 2
# Index  8: Move 2 → target ally
# Index  9: Move 3 → target opponent slot 1
# Index 10: Move 3 → target opponent slot 2
# Index 11: Move 3 → target ally
# Index 12: Switch to bench slot 0
# Index 13: Switch to bench slot 1
# Index 14: Switch to bench slot 2
# Index 15: Switch to bench slot 3
# Index 16: Struggle (no PP left)
# Index 17: Pass (empty slot or forced)
#
# Total: 18 actions per slot

N_MOVE_SLOTS = 4
N_TARGETS = 3          # opp1, opp2, ally
N_SWITCH_SLOTS = 4
N_SPECIAL = 2          # struggle, pass
N_ACTIONS = N_MOVE_SLOTS * N_TARGETS + N_SWITCH_SLOTS + N_SPECIAL  # 18

# Target position constants from DoubleBattle
# Target position constants from DoubleBattle
OPP1 = DoubleBattle.OPPONENT_1_POSITION   # 1
OPP2 = DoubleBattle.OPPONENT_2_POSITION   # 2
ALLY1 = DoubleBattle.POKEMON_1_POSITION   # -1
ALLY2 = DoubleBattle.POKEMON_2_POSITION   # -2
EMPTY_TARGET = DoubleBattle.EMPTY_TARGET_POSITION  # 0

ALLY_POSITIONS = {ALLY1, ALLY2}
OPP_POSITIONS = {OPP1, OPP2}

# Map action index → (move_slot, target_idx) for move actions
# target_idx: 0=opp1, 1=opp2, 2=ally
MOVE_ACTION_MAP = {}
for move_slot in range(N_MOVE_SLOTS):
    for target_idx in range(N_TARGETS):
        action_idx = move_slot * N_TARGETS + target_idx
        MOVE_ACTION_MAP[action_idx] = (move_slot, target_idx)

# Switch actions start at index 12
SWITCH_ACTION_START = N_MOVE_SLOTS * N_TARGETS   # 12
STRUGGLE_ACTION = SWITCH_ACTION_START + N_SWITCH_SLOTS      # 16
PASS_ACTION = STRUGGLE_ACTION + 1                           # 17


def get_action_mask(battle: DoubleBattle, slot: int) -> np.ndarray:
    """
    Build a binary action mask for a given slot.
    1 = action is valid, 0 = action is invalid.

    Args:
        battle: current DoubleBattle state
        slot:   0 or 1 (which active pokemon slot)

    Returns:
        mask: np.ndarray of shape (18,) with 0/1 values
    """
    mask = np.zeros(N_ACTIONS, dtype=np.float32)

    mon = battle.active_pokemon[slot]

    # Empty or fainted slot — only pass is valid
    if mon is None or mon.fainted:
        mask[PASS_ACTION] = 1.0
        return mask

    # Force switch — only switches are valid
    if battle.force_switch[slot]:
        switches = battle.available_switches[slot]
        bench = _get_bench_pokemon(battle)
        for i, bench_mon in enumerate(bench):
            if bench_mon is not None and bench_mon in switches:
                mask[SWITCH_ACTION_START + i] = 1.0
        # If no switches available, pass
        if mask.sum() == 0:
            mask[PASS_ACTION] = 1.0
        return mask

    # Trapped — cannot switch out
    is_trapped = False
    try:
        is_trapped = battle.trapped[slot]
    except Exception:
        pass

    available_moves = battle.available_moves[slot]
    available_move_ids = {m.id for m in available_moves}

    # Get all 4 move slots
    known_moves = list(mon.moves.values())[:4]
    while len(known_moves) < 4:
        known_moves.append(None)

    # Check which opponents are alive for targeting
    opp1_alive = (
        battle.opponent_active_pokemon[0] is not None
        and not battle.opponent_active_pokemon[0].fainted
    )
    opp2_alive = (
        battle.opponent_active_pokemon[1] is not None
        and not battle.opponent_active_pokemon[1].fainted
    )

    # Get ally (the other active slot)
    ally_slot = 1 - slot
    ally = battle.active_pokemon[ally_slot]
    ally_alive = ally is not None and not ally.fainted

    for move_slot, move in enumerate(known_moves):
        if move is None:
            continue
        if move.id not in available_move_ids:
            continue

        # Get valid showdown targets for this move
        valid_targets = battle.get_possible_showdown_targets(move, mon)

        # Check type effectiveness for immune masking
        opp1_mon = battle.opponent_active_pokemon[0]
        opp2_mon = battle.opponent_active_pokemon[1]

        opp1_multiplier = 0.0
        opp2_multiplier = 0.0
        try:
            if opp1_mon and not opp1_mon.fainted:
                opp1_multiplier = float(opp1_mon.damage_multiplier(move))
        except Exception:
            opp1_multiplier = 1.0
        try:
            if opp2_mon and not opp2_mon.fainted:
                opp2_multiplier = float(opp2_mon.damage_multiplier(move))
        except Exception:
            opp2_multiplier = 1.0

        # opp1 target — only if not immune and target is valid
        if OPP1 in valid_targets and opp1_alive and opp1_multiplier > 0.0:
            mask[move_slot * N_TARGETS + 0] = 1.0

        # opp2 target — only if not immune and target is valid
        if OPP2 in valid_targets and opp2_alive and opp2_multiplier > 0.0:
            mask[move_slot * N_TARGETS + 1] = 1.0

        # ally/self target — only allow for genuinely ally-targeting moves
        # NORMAL and ANY moves should never target allies intentionally
        from poke_env.battle.target import Target
        ALLY_ONLY_TARGETS = {
            Target.ADJACENT_ALLY,
            Target.ADJACENT_ALLY_OR_SELF,
            Target.ALLIES,
            Target.ALLY_SIDE,
            Target.ALLY_TEAM,
            Target.SELF,
        }
        is_ally_only_move = move.target in ALLY_ONLY_TARGETS
        self_target_valid = EMPTY_TARGET in valid_targets

        if is_ally_only_move and ally_alive:
            mask[move_slot * N_TARGETS + 2] = 1.0
        elif self_target_valid:
            mask[move_slot * N_TARGETS + 2] = 1.0

    # Switch actions — not available if trapped
    if not is_trapped:
        switches = battle.available_switches[slot]
        bench = _get_bench_pokemon(battle)
        for i, bench_mon in enumerate(bench):
            if bench_mon is not None and bench_mon in switches:
                mask[SWITCH_ACTION_START + i] = 1.0

    # Struggle — valid when no moves available but can still act
    if not available_moves and not switches:
        mask[STRUGGLE_ACTION] = 1.0

    # Pass — if nothing else is valid
    if mask.sum() == 0:
        mask[PASS_ACTION] = 1.0

    return mask


def _get_bench_pokemon(battle: DoubleBattle) -> List[Optional[Pokemon]]:
    """
    Returns list of 4 bench pokemon slots (non-active, non-fainted first).
    Matches the ordering used in observation.py.
    """
    active_species = {
        m.species for m in battle.active_pokemon if m is not None
    }
    bench = [
        m for m in battle.team.values()
        if m.species not in active_species and not m.fainted
    ]
    fainted = [
        m for m in battle.team.values()
        if m.fainted
    ]
    result = (bench + fainted)[:4]
    while len(result) < 4:
        result.append(None)
    return result


def action_to_order(
    action: int,
    battle: DoubleBattle,
    slot: int,
) -> SingleBattleOrder:
    """
    Convert an action index (0-17) into a SingleBattleOrder for one slot.

    Args:
        action: integer action index
        battle: current DoubleBattle
        slot:   0 or 1

    Returns:
        SingleBattleOrder for this slot
    """
    mon = battle.active_pokemon[slot]

    # Pass action
    if action == PASS_ACTION or mon is None or mon.fainted:
        return PassBattleOrder()

    # Struggle action
    if action == STRUGGLE_ACTION:
        return DefaultBattleOrder()

    # Switch action
    if action >= SWITCH_ACTION_START:
        bench_slot = action - SWITCH_ACTION_START
        bench = _get_bench_pokemon(battle)
        target_mon = bench[bench_slot]
        if target_mon is not None:
            return SingleBattleOrder(target_mon)
        return PassBattleOrder()

    # Move action
    move_slot, target_idx = MOVE_ACTION_MAP[action]
    known_moves = list(mon.moves.values())[:4]
    while len(known_moves) < 4:
        known_moves.append(None)

    move = known_moves[move_slot] if move_slot < len(known_moves) else None
    if move is None:
        return DefaultBattleOrder()

    # Resolve target position
    if target_idx == 0:
        target = OPP1
    elif target_idx == 1:
        target = OPP2
    else:
        # Ally/self target
        valid_targets = battle.get_possible_showdown_targets(move, mon)
        ally_targets = [t for t in valid_targets if t in ALLY_POSITIONS]
        if ally_targets:
            target = ally_targets[0]
        else:
            target = EMPTY_TARGET

    return SingleBattleOrder(move, move_target=target)


def actions_to_double_order(
    action_0: int,
    action_1: int,
    battle: DoubleBattle,
) -> DoubleBattleOrder:
    """
    Convert two action indices into a full DoubleBattleOrder.

    Args:
        action_0: action for slot 0
        action_1: action for slot 1
        battle:   current DoubleBattle

    Returns:
        DoubleBattleOrder ready to send to Showdown
    """
    order_0 = action_to_order(action_0, battle, slot=0)
    order_1 = action_to_order(action_1, battle, slot=1)

    joined = DoubleBattleOrder.join_orders([order_0], [order_1])
    if joined:
        return joined[0]
    return DoubleBattleOrder(order_0, DefaultBattleOrder())


def get_action_masks_tensor(
    battle: DoubleBattle,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get action masks for both slots as PyTorch tensors.
    Ready to feed directly into the policy network.

    Returns:
        mask_0: (1, 18) bool tensor for slot 0
        mask_1: (1, 18) bool tensor for slot 1
    """
    mask_0 = torch.tensor(
        get_action_mask(battle, slot=0),
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)

    mask_1 = torch.tensor(
        get_action_mask(battle, slot=1),
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)

    return mask_0, mask_1


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Action Space Layout:")
    print(f"  Total actions per slot: {N_ACTIONS}")
    print(f"  Move actions (0-11):    {N_MOVE_SLOTS} moves × {N_TARGETS} targets")
    print(f"  Switch actions (12-15): {N_SWITCH_SLOTS} bench slots")
    print(f"  Struggle (16)")
    print(f"  Pass (17)")
    print()

    print("Move action mapping:")
    targets = ["opp1", "opp2", "ally"]
    for idx in range(12):
        move_slot, target_idx = MOVE_ACTION_MAP[idx]
        print(f"  Action {idx:2d} → move slot {move_slot}, target={targets[target_idx]}")

    print()
    print("Switch action mapping:")
    for i in range(4):
        print(f"  Action {SWITCH_ACTION_START + i} → switch to bench slot {i}")

    print(f"\n  Action {STRUGGLE_ACTION} → Struggle")
    print(f"  Action {PASS_ACTION} → Pass")