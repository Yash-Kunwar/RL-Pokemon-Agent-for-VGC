import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Dict
from functools import lru_cache

from poke_env.battle.pokemon import Pokemon
from poke_env.battle.move import Move
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.target import Target

from src.data.replay_parser import (
    ParsedBattle, TurnState, TurnAction, PokemonState,
    load_battles_from_file,
)
from src.utils.observation import (
    N_IMMUNITY_FLAGS, POKEMON_TYPES, STATUSES, TRACKED_EFFECTS,
    WEATHERS, TERRAINS, FIELDS, SIDE_CONDITIONS,
    BOOST_STATS, BASE_STATS, ITEM_LIST, ITEM_TO_IDX,
    N_TYPES, N_STATUSES, N_EFFECTS, N_WEATHERS,
    N_TERRAINS, N_FIELDS, N_SIDE_CONDITIONS,
    N_BOOSTS, N_BASE_STATS, N_ITEMS,
    get_observation_size,
)
from src.utils.action_space import (
    N_ACTIONS, N_MOVE_SLOTS, N_TARGETS,
    SWITCH_ACTION_START, STRUGGLE_ACTION, PASS_ACTION,
    MOVE_ACTION_MAP,
)

# ─── poke-env data cache ───────────────────────────────────────────────────────
# Load once, reuse everywhere — expensive to initialize

_GEN9_DATA = None

def get_gen9_data():
    global _GEN9_DATA
    if _GEN9_DATA is None:
        from poke_env.data import GenData
        _GEN9_DATA = GenData.from_gen(9)
    return _GEN9_DATA


@lru_cache(maxsize=2000)
def _get_pokemon_obj(species: str) -> Optional[Pokemon]:
    """Load a Pokemon object by species name, cached."""
    try:
        return Pokemon(gen=9, species=species)
    except Exception:
        return None


@lru_cache(maxsize=5000)
def _get_move_obj(move_id: str) -> Optional[Move]:
    """Load a Move object by id, cached."""
    try:
        return Move(move_id, gen=9)
    except Exception:
        return None


# ─── Pokemon encoding ──────────────────────────────────────────────────────────

def _encode_parsed_pokemon(
    state: Optional[PokemonState],
    is_active: bool,
    hide_item: bool = False,
    hide_ability: bool = False,
) -> np.ndarray:
    """
    Encode a PokemonState into the same 231-dim vector as _encode_pokemon()
    in observation.py.

    hide_item / hide_ability: set True for opponent's pokemon to simulate
    the partial observability of real battles.
    """
    item_vec_size = N_ITEMS + 1
    tera_vec_size = N_TYPES + 1
    per_mon_size = (
        1 + N_TYPES + N_BASE_STATS + N_BOOSTS + N_STATUSES
        + N_EFFECTS + 1 + 1 + item_vec_size + tera_vec_size + 1 + N_IMMUNITY_FLAGS
    ) 

    vec = np.zeros(per_mon_size, dtype=np.float32)

    if state is None:
        # Empty slot — mark as fainted
        fainted_idx = 1 + N_TYPES + N_BASE_STATS + N_BOOSTS + N_STATUSES + N_EFFECTS + 1
        vec[fainted_idx] = 1.0
        return vec

    idx = 0

    # HP fraction
    vec[idx] = 0.0 if state.fainted else state.hp_fraction
    idx += 1

    # Load poke-env Pokemon object for static data (types, base stats)
    mon_obj = _get_pokemon_obj(state.species)

    # Types (18)
    if mon_obj is not None:
        for t in mon_obj.types:
            if t is not None and t in POKEMON_TYPES:
                vec[idx + POKEMON_TYPES.index(t)] = 1.0
    idx += N_TYPES

    # Base stats normalized by 255 (6)
    if mon_obj is not None:
        for i, stat in enumerate(BASE_STATS):
            vec[idx + i] = mon_obj.base_stats.get(stat, 0) / 255.0
    idx += N_BASE_STATS

    # Stat boosts normalized to [-1, 1] (7)
    for i, stat in enumerate(BOOST_STATS):
        vec[idx + i] = state.boosts.get(stat, 0) / 6.0
    idx += N_BOOSTS

    # Non-volatile status (7)
    if state.status is not None:
        for i, s in enumerate(STATUSES):
            if s.name == state.status:
                vec[idx + i] = 1.0
                break
    idx += N_STATUSES

    # Volatile effects (54)
    for effect_name in state.effects:
        for i, eff in enumerate(TRACKED_EFFECTS):
            if eff.name == effect_name:
                vec[idx + i] = 1.0
                break
    idx += N_EFFECTS

    # is_active (1)
    vec[idx] = 1.0 if is_active else 0.0
    idx += 1

    # is_fainted (1)
    vec[idx] = 1.0 if state.fainted else 0.0
    idx += 1

    # Item one-hot (N_ITEMS + 1)
    if not hide_item and state.item:
        item_key = state.item.lower().replace(' ', '').replace('-', '')
        if item_key in ITEM_TO_IDX:
            vec[idx + ITEM_TO_IDX[item_key]] = 1.0
        else:
            vec[idx + N_ITEMS] = 1.0  # unknown item
    else:
        vec[idx + N_ITEMS] = 1.0  # hidden or no item
    idx += item_vec_size

    # Tera type (N_TYPES + 1)
    if state.tera_type is not None:
        tera_normalized = state.tera_type.upper().replace(' ', '_')
        for i, t in enumerate(POKEMON_TYPES):
            if t.name == tera_normalized:
                vec[idx + i] = 1.0
                break
        else:
            vec[idx + N_TYPES] = 1.0  # unknown
    else:
        vec[idx + N_TYPES] = 1.0
    idx += tera_vec_size

    # is_terastallized (1)
    vec[idx] = 1.0 if state.is_terastallized else 0.0
    idx += 1

    # immunity flags (10)
    if mon_obj is not None:
        from src.utils.observation import _get_immunity_flags
        vec[idx:idx + N_IMMUNITY_FLAGS] = _get_immunity_flags(mon_obj)
    idx += N_IMMUNITY_FLAGS

    return vec


# ─── Move encoding ─────────────────────────────────────────────────────────────

def _encode_parsed_move(
    move_id: Optional[str],
    is_usable: bool,
) -> np.ndarray:
    """
    Encode a move by its ID string into the same 28-dim vector
    as _encode_move() in observation.py.
    """
    vec = np.zeros(30, dtype=np.float32)

    if move_id is None:
        return vec

    move_obj = _get_move_obj(move_id)
    if move_obj is None:
        return vec

    idx = 0

    # Base power normalized
    vec[idx] = min(move_obj.base_power / 250.0, 1.0)
    idx += 1

    # Accuracy
    acc = move_obj.accuracy
    vec[idx] = float(acc) if isinstance(acc, float) else 1.0
    idx += 1

    # Type (18)
    if move_obj.type is not None and move_obj.type in POKEMON_TYPES:
        vec[idx + POKEMON_TYPES.index(move_obj.type)] = 1.0
    idx += N_TYPES

    # Category (3)
    cat_map = {
        MoveCategory.PHYSICAL: 0,
        MoveCategory.SPECIAL: 1,
        MoveCategory.STATUS: 2,
    }
    if move_obj.category in cat_map:
        vec[idx + cat_map[move_obj.category]] = 1.0
    idx += 3

    # is_usable (1)
    vec[idx] = 1.0 if is_usable else 0.0
    idx += 1

    # PP fraction — assume full PP in replay context
    vec[idx] = 1.0
    idx += 1

    # Makes contact
    vec[idx] = 1.0 if 'contact' in move_obj.flags else 0.0
    idx += 1

    # Is spread move
    spread_names = {"ALL_ADJACENT_FOES", "ALL_ADJACENT", "ALL"}
    vec[idx] = 1.0 if move_obj.target.name in spread_names else 0.0
    idx += 1

    # Priority
    vec[idx] = move_obj.priority / 7.0
    idx += 1

    # Damage oracle placeholders — 1.0 neutral (full info not available at parse time)
    vec[idx] = 0.25      # vs_opp1 normalized neutral
    idx += 1
    vec[idx] = 0.25      # vs_opp2 normalized neutral
    idx += 1

    return vec


# ─── Global state encoding ─────────────────────────────────────────────────────

WEATHER_NAME_MAP = {
    'sunnyday': 'SUNNYDAY', 'desolateland': 'DESOLATELAND',
    'raindance': 'RAINDANCE', 'primordialsea': 'PRIMORDIALSEA',
    'sandstorm': 'SANDSTORM', 'snowscape': 'SNOWSCAPE',
    'hail': 'HAIL', 'deltastream': 'DELTASTREAM',
}

TERRAIN_NAME_MAP = {
    'electricterrain': 'ELECTRIC_TERRAIN',
    'grassyterrain': 'GRASSY_TERRAIN',
    'mistyterrain': 'MISTY_TERRAIN',
    'psychicterrain': 'PSYCHIC_TERRAIN',
}

FIELD_NAME_MAP = {
    'trickroom': 'TRICK_ROOM',
    'gravity': 'GRAVITY',
    'magicroom': 'MAGIC_ROOM',
    'wonderroom': 'WONDER_ROOM',
    'fairylock': 'FAIRY_LOCK',
}

SIDE_CONDITION_NAME_MAP = {
    'tailwind': 'TAILWIND',
    'reflect': 'REFLECT',
    'lightscreen': 'LIGHT_SCREEN',
    'auroraveil': 'AURORA_VEIL',
    'safeguard': 'SAFEGUARD',
    'mist': 'MIST',
    'quickguard': 'QUICK_GUARD',
    'wideguard': 'WIDE_GUARD',
    'craftyshield': 'CRAFTY_SHIELD',
    'stickyweb': 'STICKY_WEB',
    'stealthrock': 'STEALTH_ROCK',
    'spikes': 'SPIKES',
    'toxicspikes': 'TOXIC_SPIKES',
}


def _encode_global_state(
    state: TurnState,
    perspective: str,  # 'p1' or 'p2'
) -> np.ndarray:
    """
    Encode global battle state into the same 47-dim vector as embed_battle().
    perspective determines which side conditions are 'ours' vs 'opponent's'.
    """
    size = N_WEATHERS + N_TERRAINS + N_FIELDS + 2 * N_SIDE_CONDITIONS + 1
    vec = np.zeros(size, dtype=np.float32)
    g_idx = 0

    # Weather (8)
    if state.weather:
        w_normalized = state.weather.lower().replace(' ', '').replace('-', '')
        w_name = WEATHER_NAME_MAP.get(w_normalized)
        if w_name:
            for i, w in enumerate(WEATHERS):
                if w.name == w_name:
                    vec[g_idx + i] = 1.0
                    break
    g_idx += N_WEATHERS

    # Terrain (4)
    for field_str in state.fields:
        f_normalized = field_str.lower().replace(' ', '').replace('-', '')
        t_name = TERRAIN_NAME_MAP.get(f_normalized)
        if t_name:
            for i, t in enumerate(TERRAINS):
                if t.name == t_name:
                    vec[g_idx + i] = 1.0
                    break
    g_idx += N_TERRAINS

    # Other fields: Trick Room, Gravity etc. (8)
    for field_str in state.fields:
        f_normalized = field_str.lower().replace(' ', '').replace('-', '')
        f_name = FIELD_NAME_MAP.get(f_normalized)
        if f_name:
            for i, f in enumerate(FIELDS):
                if f.name == f_name:
                    vec[g_idx + i] = 1.0
                    break
    g_idx += N_FIELDS

    # Our side conditions (13)
    our_side = state.p1_side if perspective == 'p1' else state.p2_side
    for sc_str in our_side:
        sc_normalized = sc_str.lower().replace(' ', '').replace('-', '')
        sc_name = SIDE_CONDITION_NAME_MAP.get(sc_normalized)
        if sc_name:
            for i, sc in enumerate(SIDE_CONDITIONS):
                if sc.name == sc_name:
                    vec[g_idx + i] = 1.0
                    break
    g_idx += N_SIDE_CONDITIONS

    # Opponent side conditions (13)
    opp_side = state.p2_side if perspective == 'p1' else state.p1_side
    for sc_str in opp_side:
        sc_normalized = sc_str.lower().replace(' ', '').replace('-', '')
        sc_name = SIDE_CONDITION_NAME_MAP.get(sc_normalized)
        if sc_name:
            for i, sc in enumerate(SIDE_CONDITIONS):
                if sc.name == sc_name:
                    vec[g_idx + i] = 1.0
                    break
    g_idx += N_SIDE_CONDITIONS

    # Turn number normalized
    vec[g_idx] = min(state.turn / 100.0, 1.0)

    return vec


# ─── Action label conversion ───────────────────────────────────────────────────

def _action_tuple_to_label(
    action: Optional[tuple],
    slot: int,
    state: TurnState,
    perspective: str,
) -> int:
    """
    Convert a parsed action tuple to an integer label (0-17).

    action format:
        ('move', move_id, target_str)  — target_str: 'opp1','opp2','self','ally'
        ('switch', species)
        None
    """
    if action is None:
        return PASS_ACTION

    action_type = action[0]

    if action_type == 'switch':
        species = action[1]
        # Find which bench slot this species is in
        our_team = state.p1_team if perspective == 'p1' else state.p2_team
        our_active = state.p1_active if perspective == 'p1' else state.p2_active
        active_species = set(s for s in our_active if s is not None)

        bench = [
            mon for mon in our_team.values()
            if mon.species not in active_species and not mon.fainted
        ]
        fainted = [
            mon for mon in our_team.values()
            if mon.fainted
        ]
        bench_order = (bench + fainted)[:4]

        for i, mon in enumerate(bench_order):
            if mon.species == species:
                return SWITCH_ACTION_START + i

        return PASS_ACTION  # fallback

    if action_type == 'move':
        move_id = action[1]
        target_str = action[2] if len(action) > 2 else 'opp1'

        # Find which move slot this move is in
        our_active = state.p1_active if perspective == 'p1' else state.p2_active
        our_team = state.p1_team if perspective == 'p1' else state.p2_team

        active_species = our_active[slot]
        if active_species is None or active_species not in our_team:
            return PASS_ACTION

        mon_state = our_team[active_species]
        known_moves = mon_state.moves[:4]
        while len(known_moves) < 4:
            known_moves.append(None)

        move_slot = None
        move_normalized = move_id.lower().replace(' ', '').replace('-', '')
        for i, m in enumerate(known_moves):
            if m is not None:
                m_normalized = m.lower().replace(' ', '').replace('-', '')
                if m_normalized == move_normalized:
                    move_slot = i
                    break

        if move_slot is None:
            return PASS_ACTION

        # Map target string to target index
        if target_str in ('opp2',):
            target_idx = 1
        elif target_str in ('ally', 'self'):
            target_idx = 2
        else:
            target_idx = 0  # opp1 default

        return move_slot * N_TARGETS + target_idx

    return PASS_ACTION


# ─── Full observation from TurnState ──────────────────────────────────────────

def embed_turn_state(
    state: TurnState,
    perspective: str,  # 'p1' or 'p2'
) -> np.ndarray:
    """
    Convert a TurnState into the same 2923-dim observation vector
    as embed_battle() in observation.py.

    perspective: which player's POV to encode from.
    For opponent pokemon, items/abilities are hidden to simulate
    real battle partial observability.
    """
    parts = []

    our_team = state.p1_team if perspective == 'p1' else state.p2_team
    opp_team = state.p2_team if perspective == 'p1' else state.p1_team
    our_active = state.p1_active if perspective == 'p1' else state.p2_active
    opp_active = state.p2_active if perspective == 'p1' else state.p1_active

    # ── Our active slots (2) ──────────────────────────────────────────────────
    for species in our_active:
        mon_state = our_team.get(species) if species else None
        parts.append(_encode_parsed_pokemon(mon_state, is_active=True, hide_item=False))

    # ── Our bench slots (4) ───────────────────────────────────────────────────
    active_set = set(s for s in our_active if s is not None)
    bench = [m for m in our_team.values() if m.species not in active_set and not m.fainted]
    fainted = [m for m in our_team.values() if m.fainted]
    bench_slots = (bench + fainted)[:4]
    while len(bench_slots) < 4:
        bench_slots.append(None)
    for mon_state in bench_slots:
        parts.append(_encode_parsed_pokemon(mon_state, is_active=False, hide_item=False))

    # ── Opponent active slots (2) — hide items ────────────────────────────────
    for species in opp_active:
        mon_state = opp_team.get(species) if species else None
        parts.append(_encode_parsed_pokemon(
            mon_state, is_active=True,
            hide_item=True,    # simulate partial observability
            hide_ability=True,
        ))

    # ── Opponent bench slots (4) — hide items ─────────────────────────────────
    opp_active_set = set(s for s in opp_active if s is not None)
    opp_bench = [m for m in opp_team.values() if m.species not in opp_active_set and not m.fainted]
    opp_fainted = [m for m in opp_team.values() if m.fainted]
    opp_bench_slots = (opp_bench + opp_fainted)[:4]
    while len(opp_bench_slots) < 4:
        opp_bench_slots.append(None)
    for mon_state in opp_bench_slots:
        parts.append(_encode_parsed_pokemon(
            mon_state, is_active=False,
            hide_item=True,
            hide_ability=True,
        ))

    # ── Our active slots' moves (2 slots × 4 moves) ───────────────────────────
    for species in our_active:
        mon_state = our_team.get(species) if species else None
        if mon_state is None:
            for _ in range(4):
                parts.append(np.zeros(28, dtype=np.float32))
            continue

        known_moves = mon_state.moves[:4]
        while len(known_moves) < 4:
            known_moves.append(None)

        for move_id in known_moves:
            parts.append(_encode_parsed_move(move_id, is_usable=True))

    # ── Global state ──────────────────────────────────────────────────────────
    parts.append(_encode_global_state(state, perspective))

    obs = np.concatenate(parts, axis=0)
    return obs


# ─── PyTorch Dataset ───────────────────────────────────────────────────────────

class VGCBattleDataset(Dataset):
    """
    PyTorch Dataset for Behavioral Cloning.

    Each sample is:
        obs:      float32 tensor of shape (2923,)
        label_0:  int64 — action label for slot 0 (0-17)
        label_1:  int64 — action label for slot 1 (0-17)

    Both players' perspectives are included as separate samples.
    """

    def __init__(
        self,
        battles: List[ParsedBattle],
        skip_pass_only: bool = True,
    ):
        """
        Args:
            battles:        list of ParsedBattle objects
            skip_pass_only: skip turns where both slots are PASS
                            (uninformative for training)
        """
        self.samples = []
        self._build_samples(battles, skip_pass_only)
        print(f"Dataset built: {len(self.samples)} samples")

    def _build_samples(
        self,
        battles: List[ParsedBattle],
        skip_pass_only: bool,
    ):
        skipped = 0
        errors = 0

        for battle in battles:
            for state, p1_action, p2_action in battle.turns:
                # Add p1 perspective
                try:
                    label_0_p1 = _action_tuple_to_label(
                        p1_action.slot0, 0, state, 'p1'
                    )
                    label_1_p1 = _action_tuple_to_label(
                        p1_action.slot1, 1, state, 'p1'
                    )
                    if skip_pass_only and label_0_p1 == PASS_ACTION and label_1_p1 == PASS_ACTION:
                        skipped += 1
                    else:
                        self.samples.append(('p1', state, label_0_p1, label_1_p1))
                except Exception:
                    errors += 1

                # Add p2 perspective
                try:
                    label_0_p2 = _action_tuple_to_label(
                        p2_action.slot0, 0, state, 'p2'
                    )
                    label_1_p2 = _action_tuple_to_label(
                        p2_action.slot1, 1, state, 'p2'
                    )
                    if skip_pass_only and label_0_p2 == PASS_ACTION and label_1_p2 == PASS_ACTION:
                        skipped += 1
                    else:
                        self.samples.append(('p2', state, label_0_p2, label_1_p2))
                except Exception:
                    errors += 1

        print(f"  Skipped {skipped} pass-only turns, {errors} errors")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        perspective, state, label_0, label_1 = self.samples[idx]

        obs = embed_turn_state(state, perspective)
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        label_0_tensor = torch.tensor(label_0, dtype=torch.long)
        label_1_tensor = torch.tensor(label_1, dtype=torch.long)

        return obs_tensor, label_0_tensor, label_1_tensor


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    print("Loading battles...")
    battles = load_battles_from_file(
        'data/replays/logs_gen9vgc2025regi.json',
        max_battles=200,
    )

    print("\nBuilding dataset...")
    dataset = VGCBattleDataset(battles)

    print(f"\nDataset size: {len(dataset)}")

    # Test single sample
    obs, label_0, label_1 = dataset[0]
    print(f"\nSample 0:")
    print(f"  obs shape:   {obs.shape}  (expected: {get_observation_size()})")
    print(f"  label_0:     {label_0.item()}  (range: 0-{N_ACTIONS-1})")
    print(f"  label_1:     {label_1.item()}  (range: 0-{N_ACTIONS-1})")
    print(f"  obs has NaN: {torch.isnan(obs).any().item()}")
    print(f"  obs has Inf: {torch.isinf(obs).any().item()}")

    # Test DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    batch_obs, batch_l0, batch_l1 = next(iter(loader))
    print(f"\nBatch test:")
    print(f"  batch obs shape:    {batch_obs.shape}")
    print(f"  batch label_0 shape:{batch_l0.shape}")
    print(f"  batch label_1 shape:{batch_l1.shape}")

    # Action distribution
    all_labels = [dataset[i][1].item() for i in range(min(1000, len(dataset)))]
    from collections import Counter
    counts = Counter(all_labels)
    print(f"\nAction distribution (slot 0, first 1000 samples):")
    targets = ['opp1', 'opp2', 'ally']
    for action_idx in sorted(counts.keys()):
        count = counts[action_idx]
        if action_idx < 12:
            move_slot, target_idx = action_idx // 3, action_idx % 3
            label = f"move{move_slot}->{targets[target_idx]}"
        elif action_idx < 16:
            label = f"switch{action_idx - 12}"
        elif action_idx == 16:
            label = "struggle"
        else:
            label = "pass"
        print(f"  {action_idx:2d} ({label:20s}): {count}")