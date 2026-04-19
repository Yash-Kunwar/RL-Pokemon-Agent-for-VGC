import numpy as np
from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.move import Move
from poke_env.battle.weather import Weather
from poke_env.battle.field import Field
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.effect import Effect
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon_type import PokemonType
from typing import Optional, List


# ─── Constants ────────────────────────────────────────────────────────────────

POKEMON_TYPES = [
    PokemonType.BUG, PokemonType.DARK, PokemonType.DRAGON, PokemonType.ELECTRIC,
    PokemonType.FAIRY, PokemonType.FIGHTING, PokemonType.FIRE, PokemonType.FLYING,
    PokemonType.GHOST, PokemonType.GRASS, PokemonType.GROUND, PokemonType.ICE,
    PokemonType.NORMAL, PokemonType.POISON, PokemonType.PSYCHIC, PokemonType.ROCK,
    PokemonType.STEEL, PokemonType.WATER,
]  # 18 types (excluding THREE_QUESTION_MARKS and STELLAR — not real combat types)

STATUSES = [
    Status.BRN, Status.FNT, Status.FRZ, Status.PAR,
    Status.PSN, Status.SLP, Status.TOX,
]  # 7 non-volatile statuses

# Volatile effects we care about for decision making
TRACKED_EFFECTS = [
    Effect.CONFUSION,
    Effect.LEECH_SEED,
    Effect.SALT_CURE,
    Effect.INFESTATION,
    Effect.BIND,
    Effect.WRAP,
    Effect.CLAMP,
    Effect.FIRE_SPIN,
    Effect.WHIRLPOOL,
    Effect.MAGMA_STORM,
    Effect.SAND_TOMB,
    Effect.SNAP_TRAP,
    Effect.THUNDER_CAGE,
    Effect.PARTIALLY_TRAPPED,  # generic trapping
    Effect.TAUNT,
    Effect.ENCORE,
    Effect.FLINCH,
    Effect.YAWN,
    Effect.PERISH0,
    Effect.PERISH1,
    Effect.PERISH2,
    Effect.PERISH3,
    Effect.CURSE,
    Effect.SUBSTITUTE,
    Effect.HELPING_HAND,
    Effect.FOLLOW_ME,
    Effect.RAGE_POWDER,
    Effect.DESTINY_BOND,
    Effect.PROTECT,
    Effect.BANEFUL_BUNKER,
    Effect.SPIKY_SHIELD,
    Effect.SILK_TRAP,
    Effect.BURNING_BULWARK,
    Effect.KINGS_SHIELD,
    Effect.OCTOLOCK,
    Effect.NO_RETREAT,
    Effect.INGRAIN,
    Effect.TRAPPED,
    Effect.SYRUP_BOMB,
    Effect.ELECTRIFY,
    Effect.THROAT_CHOP,
    Effect.TORMENT,
    Effect.DISABLE,
    Effect.HEAL_BLOCK,
    Effect.EMBARGO,
    Effect.TELEKINESIS,
    Effect.SMACK_DOWN,
    Effect.SPOTLIGHT,
    Effect.INSTRUCT,
    Effect.QUASH,
    Effect.AFTER_YOU,
    Effect.WIDE_GUARD,
    Effect.QUICK_GUARD,
    Effect.CRAFTY_SHIELD,
    Effect.MAT_BLOCK,
]  # 54 volatile effects

WEATHERS = [
    Weather.SUNNYDAY, Weather.DESOLATELAND,
    Weather.RAINDANCE, Weather.PRIMORDIALSEA,
    Weather.SANDSTORM, Weather.SNOWSCAPE,
    Weather.HAIL, Weather.DELTASTREAM,
]  # 8 weather states (UNKNOWN = none active)

TERRAINS = [
    Field.ELECTRIC_TERRAIN,
    Field.GRASSY_TERRAIN,
    Field.MISTY_TERRAIN,
    Field.PSYCHIC_TERRAIN,
]  # 4 terrains

FIELDS = [
    Field.TRICK_ROOM,
    Field.GRAVITY,
    Field.MAGIC_ROOM,
    Field.WONDER_ROOM,
    Field.FAIRY_LOCK,
    Field.HEAL_BLOCK,
    Field.MUD_SPORT,
    Field.WATER_SPORT,
]  # 8 other global fields

SIDE_CONDITIONS = [
    SideCondition.TAILWIND,
    SideCondition.REFLECT,
    SideCondition.LIGHT_SCREEN,
    SideCondition.AURORA_VEIL,
    SideCondition.SAFEGUARD,
    SideCondition.MIST,
    SideCondition.QUICK_GUARD,
    SideCondition.WIDE_GUARD,
    SideCondition.CRAFTY_SHIELD,
    SideCondition.STICKY_WEB,
    SideCondition.STEALTH_ROCK,
    SideCondition.SPIKES,
    SideCondition.TOXIC_SPIKES,
]  # 13 side conditions per side

BOOST_STATS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]  # 7 boosts

BASE_STATS = ["hp", "atk", "def", "spa", "spd", "spe"]  # 6 base stats

# Competitive items grouped by effect category
ITEM_LIST = [
    # Choice items
    "choicescarf", "choiceband", "choicespecs",
    # Berries
    "sitrusberry", "lumberry", "leppaberry", "oranberry", "figyberry",
    "wikiberry", "magoberry", "aguavberry", "iapapaberry", "custapberry",
    "enigmaberry", "rowapberry", "jabocaberry", "keeberry", "marangaberry",
    "salacberry", "petayaberry", "apicotberry", "liechiberry", "ganlonberry",
    "starfberry", "micleberry", "lansat berry", "occaberry", "passhoberry",
    "wacanberry", "rindoberry", "yacheberry", "choplberry", "kebiaberry",
    "shucaberry", "cobaberry", "payapaberry", "tangaberry", "chartiberry",
    "kasibberry", "habanberry", "colburberry", "babiriberry", "chilanberry",
    "roseliberry", "loudberry",
    # Assault / defensive
    "assaultvest", "eviolite", "rockyhelmet", "roughskin",
    # Terrain / weather seeds
    "electricseed", "grassyseed", "mistyseed", "psychicseed",
    "roomservice", "utilityumbrella",
    # Speed control
    "ironball",
    # HP / recovery
    "leftovers", "blacksludge", "shellbell",
    # Damage boost
    "lifeorb", "expertbelt", "wiseglasses", "muscleband",
    "punchingglove", "throatspray",
    # Focus
    "focussash", "focusband",
    # Redirection / misc
    "mentalherb", "whiteherb", "powerherb", "redcard",
    "ejectbutton", "ejectpack", "shedshell",
    "heavydutyboots", "clearamulet", "covertcloak",
    "mirrorherb", "loadeddice", "boosterenergy",
    # Mega stones (legacy but still parsed)
    "blueorb", "redorb",
    # Type-boosting
    "charcoal", "mysticwater", "miracleseed", "nevermeltice",
    "magnet", "twistedspoon", "blackbelt", "poisonbarb", "softsan",
    "spelltag", "metalcoat", "silkscarf", "sharpbeak", "blackglasses",
    "dragonscale", "pixieplate", "fairyfeather",
    # Z-crystals (legacy)
    "normaliumz", "firiumz", "wateriumz", "electriumz", "grassiumz",
]  # ~120 items

N_TYPES = len(POKEMON_TYPES)          # 18
N_STATUSES = len(STATUSES)            # 7
N_EFFECTS = len(TRACKED_EFFECTS)      # 54
N_WEATHERS = len(WEATHERS)            # 8
N_TERRAINS = len(TERRAINS)            # 4
N_FIELDS = len(FIELDS)                # 8
N_SIDE_CONDITIONS = len(SIDE_CONDITIONS)  # 13
N_BOOSTS = len(BOOST_STATS)           # 7
N_BASE_STATS = len(BASE_STATS)        # 6
N_ITEMS = len(ITEM_LIST)              # ~120
N_MOVE_CATEGORIES = 3                 # physical, special, status

ITEM_TO_IDX = {item: i for i, item in enumerate(ITEM_LIST)}


# ─── Dimension calculation ─────────────────────────────────────────────────────
#
# Per Pokémon (12 total: 4 active + 8 bench):
#   hp_fraction          : 1
#   types                : 18
#   base_stats           : 6
#   boosts               : 7  (only meaningful for active mons, 0 for bench)
#   status               : 7
#   volatile_effects     : 54
#   is_active            : 1
#   is_fainted           : 1
#   item                 : N_ITEMS + 1 (one-hot + unknown slot)
#   tera_type            : 18 + 1 (one-hot + unknown)
#   is_terastallized     : 1
#   ---------------------
#   subtotal             : 1+18+6+7+7+54+1+1+(N_ITEMS+1)+(19)+1 = 116 + N_ITEMS
#
# Per active Pokémon's moves (2 active mons × 4 moves = 8 move slots):
#   base_power_norm      : 1
#   accuracy             : 1
#   type                 : 18
#   category             : 3
#   is_usable            : 1
#   pp_fraction          : 1
#   makes_contact        : 1
#   is_spread            : 1   (hits multiple targets: Earthquake, Surf etc.)
#   priority             : 1
#   ---------------------
#   subtotal             : 28
#
# Global state:
#   weather              : 8
#   terrain              : 4
#   other_fields         : 8
#   our_side_conditions  : 13
#   opp_side_conditions  : 13
#   turn_number_norm     : 1
#   ---------------------
#   subtotal             : 47
#
# TOTAL = 12*(116+N_ITEMS) + 8*28 + 47


def _encode_pokemon(mon: Optional[Pokemon], is_active: bool) -> np.ndarray:
    """Encode a single Pokémon into a fixed-size float32 vector."""
    item_vec_size = N_ITEMS + 1   # +1 for unknown/no item
    tera_vec_size = N_TYPES + 1   # +1 for unknown tera type
    per_mon_size = 1 + N_TYPES + N_BASE_STATS + N_BOOSTS + N_STATUSES + N_EFFECTS + 1 + 1 + item_vec_size + tera_vec_size + 1

    vec = np.zeros(per_mon_size, dtype=np.float32)

    if mon is None:
        # Empty slot — all zeros is fine, is_fainted=1 to signal unavailability
        vec[1 + N_TYPES + N_BASE_STATS + N_BOOSTS + N_STATUSES + N_EFFECTS + 1] = 1.0
        return vec

    idx = 0

    # HP fraction [0, 1]
    vec[idx] = mon.current_hp_fraction if not mon.fainted else 0.0
    idx += 1

    # Types — multi-hot (18)
    for t in mon.types:
        if t is not None and t in POKEMON_TYPES:
            vec[idx + POKEMON_TYPES.index(t)] = 1.0
    idx += N_TYPES

    # Base stats normalized by 255 (max possible base stat) (6)
    for i, stat in enumerate(BASE_STATS):
        vec[idx + i] = mon.base_stats.get(stat, 0) / 255.0
    idx += N_BASE_STATS

    # Stat boosts normalized to [-1, 1] from [-6, +6] (7)
    for i, stat in enumerate(BOOST_STATS):
        vec[idx + i] = mon.boosts.get(stat, 0) / 6.0
    idx += N_BOOSTS

    # Non-volatile status (7)
    if mon.status is not None:
        try:
            vec[idx + STATUSES.index(mon.status)] = 1.0
        except ValueError:
            pass
    idx += N_STATUSES

    # Volatile effects (54)
    for effect in mon.effects:
        if effect in TRACKED_EFFECTS:
            vec[idx + TRACKED_EFFECTS.index(effect)] = 1.0
    idx += N_EFFECTS

    # is_active (1)
    vec[idx] = 1.0 if is_active else 0.0
    idx += 1

    # is_fainted (1)
    vec[idx] = 1.0 if mon.fainted else 0.0
    idx += 1

    # Item — one-hot + unknown slot (N_ITEMS + 1)
    if mon.item is not None and mon.item != "":
        item_key = mon.item.lower().replace(" ", "").replace("-", "")
        if item_key in ITEM_TO_IDX:
            vec[idx + ITEM_TO_IDX[item_key]] = 1.0
        else:
            vec[idx + N_ITEMS] = 1.0  # unknown item slot
    else:
        vec[idx + N_ITEMS] = 1.0  # no item / unknown
    idx += item_vec_size

    # Tera type — one-hot + unknown (N_TYPES + 1)
    if mon.tera_type is not None and mon.tera_type in POKEMON_TYPES:
        vec[idx + POKEMON_TYPES.index(mon.tera_type)] = 1.0
    else:
        vec[idx + N_TYPES] = 1.0  # unknown
    idx += tera_vec_size

    # is_terastallized (1)
    vec[idx] = 1.0 if mon.is_terastallized else 0.0
    idx += 1

    return vec


def _encode_move(move: Optional[Move], is_usable: bool) -> np.ndarray:
    """Encode a single move into a fixed-size float32 vector."""
    per_move_size = 9 + N_TYPES  # 9 scalars + 18 type flags = 27... wait, see below
    # base_power(1) + accuracy(1) + type(18) + category(3) + is_usable(1) + pp_fraction(1) + makes_contact(1) + is_spread(1) + priority(1) = 28
    per_move_size = 28
    vec = np.zeros(per_move_size, dtype=np.float32)

    if move is None:
        return vec

    idx = 0

    # Base power normalized by 250 (highest base powers are ~250)
    vec[idx] = min(move.base_power / 250.0, 1.0)
    idx += 1

    # Accuracy [0, 1] — bypass moves have accuracy 0 in poke-env (treat as 1.0)
    acc = move.accuracy
    vec[idx] = float(acc) if isinstance(acc, float) else 1.0
    idx += 1

    # Move type — one-hot (18)
    if move.type is not None and move.type in POKEMON_TYPES:
        vec[idx + POKEMON_TYPES.index(move.type)] = 1.0
    idx += N_TYPES

    # Move category — one-hot (3)
    cat_map = {MoveCategory.PHYSICAL: 0, MoveCategory.SPECIAL: 1, MoveCategory.STATUS: 2}
    if move.category in cat_map:
        vec[idx + cat_map[move.category]] = 1.0
    idx += N_MOVE_CATEGORIES

    # is_usable (1)
    vec[idx] = 1.0 if is_usable else 0.0
    idx += 1

    # PP fraction (1)
    if move.max_pp and move.max_pp > 0:
        vec[idx] = move.current_pp / move.max_pp
    else:
        vec[idx] = 1.0
    idx += 1

    # Makes contact (1)
    vec[idx] = 1.0 if 'contact' in move.flags else 0.0
    idx += 1

    # Is spread move — hits multiple targets like Earthquake, Surf (1)
    from poke_env.battle.target import Target
    spread_names = {"ALL_ADJACENT_FOES", "ALL_ADJACENT", "ALL"}
    vec[idx] = 1.0 if move.target.name in spread_names else 0.0
    idx += 1

    # Priority normalized — range typically [-7, +5], normalize to [-1, 1]
    vec[idx] = move.priority / 7.0
    idx += 1

    return vec


def _encode_side_conditions(conditions: dict) -> np.ndarray:
    """Encode side conditions as a binary vector."""
    vec = np.zeros(N_SIDE_CONDITIONS, dtype=np.float32)
    for sc, count in conditions.items():
        if sc in SIDE_CONDITIONS:
            vec[SIDE_CONDITIONS.index(sc)] = min(count / 3.0, 1.0)  # normalize layers
    return vec


def embed_battle(battle: DoubleBattle) -> np.ndarray:
    """
    Convert a DoubleBattle into a flat float32 observation vector.

    Structure:
        - 12 Pokémon vectors (4 active + 8 bench, ally first then opponent)
        - 8 move vectors per active slot (2 slots × 4 moves)
        - Global state vector (weather, terrain, fields, side conditions, turn)
    """
    parts = []

    # ── Ally Pokémon ──────────────────────────────────────────────────────────
    active_ally = list(battle.active_pokemon)  # [mon_slot0, mon_slot1]
    active_ally_species = {m.species for m in active_ally if m is not None}

    # Encode 2 active ally slots
    for mon in active_ally:
        parts.append(_encode_pokemon(mon, is_active=True))

    # Encode up to 4 bench ally slots (skip active ones)
    bench_ally = [
        m for m in battle.team.values()
        if m.species not in active_ally_species and not m.fainted
    ]
    fainted_ally = [
        m for m in battle.team.values()
        if m.fainted
    ]
    bench_slots = (bench_ally + fainted_ally)[:4]
    while len(bench_slots) < 4:
        bench_slots.append(None)
    for mon in bench_slots:
        parts.append(_encode_pokemon(mon, is_active=False))

    # ── Opponent Pokémon ──────────────────────────────────────────────────────
    active_opp = list(battle.opponent_active_pokemon)  # [mon_slot0, mon_slot1]
    active_opp_species = {m.species for m in active_opp if m is not None}

    # Encode 2 active opponent slots
    for mon in active_opp:
        parts.append(_encode_pokemon(mon, is_active=True))

    # Encode up to 4 bench opponent slots
    bench_opp = [
        m for m in battle.opponent_team.values()
        if m.species not in active_opp_species and not m.fainted
    ]
    fainted_opp = [
        m for m in battle.opponent_team.values()
        if m.fainted
    ]
    bench_opp_slots = (bench_opp + fainted_opp)[:4]
    while len(bench_opp_slots) < 4:
        bench_opp_slots.append(None)
    for mon in bench_opp_slots:
        parts.append(_encode_pokemon(mon, is_active=False))

    # ── Move vectors (2 active ally slots × 4 moves each) ────────────────────
    for slot_idx, mon in enumerate(active_ally):
        if mon is None:
            # Empty slot — 4 zero move vectors
            for _ in range(4):
                parts.append(np.zeros(28, dtype=np.float32))
            continue

        available = battle.available_moves[slot_idx]
        available_ids = {m.id for m in available}

        # Get all 4 moves the mon knows (from mon.moves dict)
        known_moves = list(mon.moves.values())

        # Pad to 4 slots
        while len(known_moves) < 4:
            known_moves.append(None)
        known_moves = known_moves[:4]

        for move in known_moves:
            is_usable = move is not None and move.id in available_ids
            parts.append(_encode_move(move, is_usable))

    # ── Global battle state ───────────────────────────────────────────────────
    global_vec = np.zeros(N_WEATHERS + N_TERRAINS + N_FIELDS + 2 * N_SIDE_CONDITIONS + 1, dtype=np.float32)
    g_idx = 0

    # Weather (8)
    for w, _ in battle.weather.items():
        if w in WEATHERS:
            global_vec[g_idx + WEATHERS.index(w)] = 1.0
    g_idx += N_WEATHERS

    # Terrain (4)
    for f, _ in battle.fields.items():
        if f in TERRAINS:
            global_vec[g_idx + TERRAINS.index(f)] = 1.0
    g_idx += N_TERRAINS

    # Other fields: Trick Room, Gravity, etc. (8)
    for f, _ in battle.fields.items():
        if f in FIELDS:
            global_vec[g_idx + FIELDS.index(f)] = 1.0
    g_idx += N_FIELDS

    # Our side conditions (13)
    global_vec[g_idx:g_idx + N_SIDE_CONDITIONS] = _encode_side_conditions(battle.side_conditions)
    g_idx += N_SIDE_CONDITIONS

    # Opponent side conditions (13)
    global_vec[g_idx:g_idx + N_SIDE_CONDITIONS] = _encode_side_conditions(battle.opponent_side_conditions)
    g_idx += N_SIDE_CONDITIONS

    # Turn number normalized (1) — cap at turn 100
    global_vec[g_idx] = min(battle.turn / 100.0, 1.0)
    g_idx += 1

    parts.append(global_vec)

    # ── Concatenate everything ────────────────────────────────────────────────
    obs = np.concatenate(parts, axis=0)
    return obs


def get_observation_size(n_items: int = N_ITEMS) -> int:
    """Returns the total size of the observation vector."""
    per_mon = 1 + N_TYPES + N_BASE_STATS + N_BOOSTS + N_STATUSES + N_EFFECTS + 1 + 1 + (n_items + 1) + (N_TYPES + 1) + 1
    per_move = 28
    global_state = N_WEATHERS + N_TERRAINS + N_FIELDS + 2 * N_SIDE_CONDITIONS + 1
    return 12 * per_mon + 8 * per_move + global_state


if __name__ == "__main__":
    print(f"Observation vector size: {get_observation_size()}")
    print(f"  Per pokemon:    {1 + N_TYPES + N_BASE_STATS + N_BOOSTS + N_STATUSES + N_EFFECTS + 1 + 1 + (N_ITEMS+1) + (N_TYPES+1) + 1}")
    print(f"  Per move:       28")
    print(f"  12 pokemon:     {12 * (1 + N_TYPES + N_BASE_STATS + N_BOOSTS + N_STATUSES + N_EFFECTS + 1 + 1 + (N_ITEMS+1) + (N_TYPES+1) + 1)}")
    print(f"  8 move slots:   {8 * 28}")
    print(f"  Global state:   {N_WEATHERS + N_TERRAINS + N_FIELDS + 2 * N_SIDE_CONDITIONS + 1}")