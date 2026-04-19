import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class PokemonState:
    """Tracks the state of one Pokémon during replay parsing."""
    species: str
    hp_fraction: float = 1.0
    fainted: bool = False
    is_active: bool = False
    slot: Optional[int] = None          # 0 or 1 if active, None if bench
    status: Optional[str] = None
    boosts: Dict[str, int] = field(default_factory=lambda: {
        "atk": 0, "def": 0, "spa": 0, "spd": 0,
        "spe": 0, "accuracy": 0, "evasion": 0
    })
    effects: List[str] = field(default_factory=list)
    item: Optional[str] = None
    ability: Optional[str] = None
    moves: List[str] = field(default_factory=list)
    tera_type: Optional[str] = None
    is_terastallized: bool = False


@dataclass
class TurnState:
    """Snapshot of battle state at the start of a turn."""
    turn: int
    p1_team: Dict[str, PokemonState]     # species -> state
    p2_team: Dict[str, PokemonState]
    p1_active: List[Optional[str]]       # [slot0_species, slot1_species]
    p2_active: List[Optional[str]]
    weather: Optional[str] = None
    fields: List[str] = field(default_factory=list)
    p1_side: List[str] = field(default_factory=list)
    p2_side: List[str] = field(default_factory=list)


@dataclass
class TurnAction:
    """Actions taken by one player during a turn."""
    turn: int
    # Each slot: ('move', move_name, target) or ('switch', species) or None
    slot0: Optional[Tuple] = None
    slot1: Optional[Tuple] = None


@dataclass
class ParsedBattle:
    """A fully parsed battle with state-action pairs."""
    battle_id: str
    p1_name: str
    p2_name: str
    winner: Optional[str]               # 'p1' or 'p2'
    format: str
    # List of (state_snapshot, p1_action, p2_action) per turn
    turns: List[Tuple[TurnState, TurnAction, TurnAction]] = field(default_factory=list)


# ─── Team Parser ──────────────────────────────────────────────────────────────

def parse_showteam(line: str) -> Tuple[str, List[PokemonState]]:
    """
    Parse a |showteam|p1|... line into a list of PokemonState objects.
    """
    # Extract player and team string correctly
    # Line format: |showteam|p1|<team_data>
    prefix_p1 = '|showteam|p1|'
    prefix_p2 = '|showteam|p2|'

    if line.startswith(prefix_p1):
        player = 'p1'
        team_str = line[len(prefix_p1):]
    elif line.startswith(prefix_p2):
        player = 'p2'
        team_str = line[len(prefix_p2):]
    else:
        return 'unknown', []

    mons = []
    for mon_str in team_str.split(']'):
        mon_str = mon_str.strip()
        if not mon_str:
            continue

        fields = mon_str.split('|')
        if len(fields) < 5:
            continue

        species = _normalize_species(fields[0])
        item = fields[2].lower().replace(' ', '').replace('-', '') if fields[2] else None
        ability = fields[3] if fields[3] else None
        moves = [
            m.lower().replace(' ', '').replace('-', '')
            for m in fields[4].split(',') if m
        ]

        # Tera type is last value in field[11] which is comma-separated
        # format: "evs,,,,,TerType" — tera is always the 6th comma-separated value
        tera_type = None
        if len(fields) > 11 and fields[11]:
            tera_parts = fields[11].split(',')
            if len(tera_parts) >= 6 and tera_parts[5]:
                tera_type = tera_parts[5].strip()

        mon = PokemonState(
            species=species,
            item=item if item else None,
            ability=ability,
            moves=moves,
            tera_type=tera_type,
        )
        mons.append(mon)

    return player, mons


def _normalize_species(species: str) -> str:
    """Normalize species name to lowercase no-space format."""
    return species.lower().strip().replace(' ', '').replace('-', '').replace("'", '')


def _normalize_mon_name(name: str) -> str:
    """
    Convert battle ident like 'p1a: Calyrex' to normalized species.
    Strips player prefix and slot indicator.
    """
    # Remove 'p1a: ' or 'p2b: ' prefix
    if ': ' in name:
        name = name.split(': ', 1)[1]
    return _normalize_species(name)


def _slot_from_ident(ident: str) -> int:
    """Extract slot (0 or 1) from ident like 'p1a' or 'p2b'."""
    if 'a' in ident[:3]:
        return 0
    return 1


def _player_from_ident(ident: str) -> str:
    """Extract player ('p1' or 'p2') from ident like 'p1a: Calyrex'."""
    return ident[:2]


# ─── Main Parser ──────────────────────────────────────────────────────────────

def parse_battle_log(battle_id: str, log: str) -> Optional[ParsedBattle]:
    """
    Parse a single battle log string into a ParsedBattle object.

    Returns None if the battle is invalid or incomplete.
    """
    lines = log.split('\n')

    # ── Extract metadata ──────────────────────────────────────────────────────
    p1_name = p2_name = winner = format_name = None
    p1_team_list: List[PokemonState] = []
    p2_team_list: List[PokemonState] = []

    for line in lines:
        if line.startswith('|player|p1|'):
            name = line.split('|')[3]
            if name:  # only update if non-empty
                p1_name = name
        elif line.startswith('|player|p2|'):
            name = line.split('|')[3]
            if name:  # only update if non-empty
                p2_name = name
        elif line.startswith('|tier|'):
            format_name = line.split('|')[2]
        elif line.startswith('|showteam|'):
            player, mons = parse_showteam(line)
            if player == 'p1':
                p1_team_list = mons
            else:
                p2_team_list = mons
        elif line.startswith('|win|'):
            winner_name = line.split('|')[2].strip()
            if winner_name == p1_name:
                winner = 'p1'
            elif winner_name == p2_name:
                winner = 'p2'

    # Skip incomplete battles
    if not p1_name or not p2_name:
        return None
    if not p1_team_list or not p2_team_list:
        return None
    if winner is None:
        return None

    # Build team dicts keyed by normalized species
    p1_team = {mon.species: mon for mon in p1_team_list}
    p2_team = {mon.species: mon for mon in p2_team_list}

    # ── Parse turn by turn ────────────────────────────────────────────────────
    battle = ParsedBattle(
        battle_id=battle_id,
        p1_name=p1_name,
        p2_name=p2_name,
        winner=winner,
        format=format_name or '',
    )

    # Mutable state tracking
    p1_active = [None, None]    # [slot0_species, slot1_species]
    p2_active = [None, None]
    current_turn = 0
    current_weather = None
    current_fields = []
    p1_side = []
    p2_side = []

    # Actions being collected for current turn
    p1_action = TurnAction(turn=0)
    p2_action = TurnAction(turn=0)

    # State snapshot at start of current turn
    turn_state_snapshot: Optional[TurnState] = None

    def make_snapshot(turn_num: int) -> TurnState:
        import copy
        return TurnState(
            turn=turn_num,
            p1_team=copy.deepcopy(p1_team),
            p2_team=copy.deepcopy(p2_team),
            p1_active=list(p1_active),
            p2_active=list(p2_active),
            weather=current_weather,
            fields=list(current_fields),
            p1_side=list(p1_side),
            p2_side=list(p2_side),
        )

    in_battle = False

    for line in lines:
        if not line.startswith('|'):
            continue

        parts = line.split('|')
        if len(parts) < 2:
            continue
        cmd = parts[1]

        # ── Battle start ──────────────────────────────────────────────────────
        if cmd == 'start':
            in_battle = True
            continue

        if not in_battle:
            continue

        # ── Turn boundary ─────────────────────────────────────────────────────
        if cmd == 'turn':
            # Save previous turn's state+actions
            if current_turn > 0 and turn_state_snapshot is not None:
                battle.turns.append((turn_state_snapshot, p1_action, p2_action))

            current_turn = int(parts[2])
            turn_state_snapshot = make_snapshot(current_turn)
            p1_action = TurnAction(turn=current_turn)
            p2_action = TurnAction(turn=current_turn)
            continue

        # ── Switch (including lead switches at turn 0) ────────────────────────
        if cmd == 'switch' or cmd == 'drag':
            if len(parts) < 3:
                continue
            ident = parts[2]          # e.g. 'p1a: Calyrex'
            player = _player_from_ident(ident)
            slot = _slot_from_ident(ident)
            species = _normalize_mon_name(ident)

            # Find the actual species in the team (handle forme variants)
            actual_species = _find_species_in_team(
                species,
                p1_team if player == 'p1' else p2_team
            )
            if actual_species is None:
                actual_species = species

            # Update active tracking
            if player == 'p1':
                p1_active[slot] = actual_species
                if actual_species in p1_team:
                    p1_team[actual_species].is_active = True
                    p1_team[actual_species].slot = slot
            else:
                p2_active[slot] = actual_species
                if actual_species in p2_team:
                    p2_team[actual_species].is_active = True
                    p2_team[actual_species].slot = slot

            # Record as action if during a turn (not forced lead)
            if current_turn > 0 and cmd == 'switch':
                action = ('switch', actual_species)
                if player == 'p1':
                    if slot == 0:
                        p1_action.slot0 = action
                    else:
                        p1_action.slot1 = action
                else:
                    if slot == 0:
                        p2_action.slot0 = action
                    else:
                        p2_action.slot1 = action
            continue

        # ── Move ──────────────────────────────────────────────────────────────
        if cmd == 'move':
            if len(parts) < 4:
                continue
            ident = parts[2]          # e.g. 'p1a: Calyrex'
            move_name = parts[3].lower().replace(' ', '').replace('-', '')
            target_ident = parts[4] if len(parts) > 4 else ''

            player = _player_from_ident(ident)
            slot = _slot_from_ident(ident)

            # Determine target
            target = _parse_target(target_ident, player)

            action = ('move', move_name, target)
            if player == 'p1':
                if slot == 0:
                    p1_action.slot0 = action
                else:
                    p1_action.slot1 = action
            else:
                if slot == 0:
                    p2_action.slot0 = action
                else:
                    p2_action.slot1 = action
            continue

        # ── Faint ─────────────────────────────────────────────────────────────
        if cmd == 'faint':
            if len(parts) < 3:
                continue
            ident = parts[2]
            player = _player_from_ident(ident)
            slot = _slot_from_ident(ident)
            species = _normalize_mon_name(ident)

            team = p1_team if player == 'p1' else p2_team
            actual = _find_species_in_team(species, team) or species
            if actual in team:
                team[actual].fainted = True
                team[actual].is_active = False
                team[actual].hp_fraction = 0.0

            if player == 'p1':
                p1_active[slot] = None
            else:
                p2_active[slot] = None
            continue

        # ── Damage / Heal ─────────────────────────────────────────────────────
        if cmd == '-damage' or cmd == '-heal':
            if len(parts) < 4:
                continue
            ident = parts[2]
            condition = parts[3]      # e.g. '50/100' or '0 fnt'
            player = _player_from_ident(ident)
            species = _normalize_mon_name(ident)
            team = p1_team if player == 'p1' else p2_team
            actual = _find_species_in_team(species, team) or species
            if actual in team:
                hp_frac = _parse_hp_fraction(condition)
                team[actual].hp_fraction = hp_frac
            continue

        # ── Status ────────────────────────────────────────────────────────────
        if cmd == '-status':
            if len(parts) < 4:
                continue
            ident = parts[2]
            status = parts[3].upper()
            player = _player_from_ident(ident)
            species = _normalize_mon_name(ident)
            team = p1_team if player == 'p1' else p2_team
            actual = _find_species_in_team(species, team) or species
            if actual in team:
                team[actual].status = status
            continue

        if cmd == '-curestatus':
            if len(parts) < 3:
                continue
            ident = parts[2]
            player = _player_from_ident(ident)
            species = _normalize_mon_name(ident)
            team = p1_team if player == 'p1' else p2_team
            actual = _find_species_in_team(species, team) or species
            if actual in team:
                team[actual].status = None
            continue

        # ── Boosts ────────────────────────────────────────────────────────────
        if cmd == '-boost' or cmd == '-unboost':
            if len(parts) < 5:
                continue
            ident = parts[2]
            stat = parts[3]
            amount = int(parts[4])
            if cmd == '-unboost':
                amount = -amount
            player = _player_from_ident(ident)
            species = _normalize_mon_name(ident)
            team = p1_team if player == 'p1' else p2_team
            actual = _find_species_in_team(species, team) or species
            if actual in team and stat in team[actual].boosts:
                team[actual].boosts[stat] = max(-6, min(6,
                    team[actual].boosts[stat] + amount))
            continue

        if cmd == '-clearboost' or cmd == '-clearallboost':
            if len(parts) < 3:
                continue
            ident = parts[2]
            player = _player_from_ident(ident)
            species = _normalize_mon_name(ident)
            team = p1_team if player == 'p1' else p2_team
            actual = _find_species_in_team(species, team) or species
            if actual in team:
                team[actual].boosts = {k: 0 for k in team[actual].boosts}
            continue

        # ── Weather ───────────────────────────────────────────────────────────
        if cmd == '-weather':
            if len(parts) < 3:
                continue
            weather = parts[2]
            current_weather = None if weather == 'none' else weather
            continue

        # ── Field ─────────────────────────────────────────────────────────────
        if cmd == '-fieldstart':
            if len(parts) < 3:
                continue
            field_name = parts[2].replace('move: ', '').replace('ability: ', '')
            if field_name not in current_fields:
                current_fields.append(field_name)
            continue

        if cmd == '-fieldend':
            if len(parts) < 3:
                continue
            field_name = parts[2].replace('move: ', '').replace('ability: ', '')
            current_fields = [f for f in current_fields if f != field_name]
            continue

        # ── Side conditions ───────────────────────────────────────────────────
        if cmd == '-sidestart':
            if len(parts) < 4:
                continue
            side = parts[2][:2]
            condition = parts[3].replace('move: ', '')
            target_list = p1_side if side == 'p1' else p2_side
            if condition not in target_list:
                target_list.append(condition)
            continue

        if cmd == '-sideend':
            if len(parts) < 4:
                continue
            side = parts[2][:2]
            condition = parts[3].replace('move: ', '')
            if side == 'p1':
                p1_side = [c for c in p1_side if c != condition]
            else:
                p2_side = [c for c in p2_side if c != condition]
            continue

        # ── Terastallize ──────────────────────────────────────────────────────
        if cmd == '-terastallize':
            if len(parts) < 4:
                continue
            ident = parts[2]
            tera_type = parts[3]
            player = _player_from_ident(ident)
            species = _normalize_mon_name(ident)
            team = p1_team if player == 'p1' else p2_team
            actual = _find_species_in_team(species, team) or species
            if actual in team:
                team[actual].is_terastallized = True
                team[actual].tera_type = tera_type
            continue

        # ── End of battle ─────────────────────────────────────────────────────
        if cmd == 'win':
            # Save last turn
            if current_turn > 0 and turn_state_snapshot is not None:
                battle.turns.append((turn_state_snapshot, p1_action, p2_action))
            break

    return battle if battle.turns else None


# ─── Helper Functions ─────────────────────────────────────────────────────────

def _find_species_in_team(
    species: str,
    team: Dict[str, PokemonState]
) -> Optional[str]:
    """
    Find a species key in the team dict, handling forme variants.
    e.g. 'calyrexshadow' might match 'calyrexshadowrider' in the team.
    """
    if species in team:
        return species
    # Try prefix match for formes
    for key in team:
        if key.startswith(species) or species.startswith(key):
            return key
    return None


def _parse_hp_fraction(condition: str) -> float:
    """Parse '50/100' or '0 fnt' into a float [0, 1]."""
    condition = condition.strip()
    if 'fnt' in condition:
        return 0.0
    match = re.match(r'(\d+)/(\d+)', condition)
    if match:
        current = int(match.group(1))
        maximum = int(match.group(2))
        return current / maximum if maximum > 0 else 0.0
    return 1.0


def _parse_target(target_ident: str, acting_player: str) -> str:
    """
    Convert target ident like 'p2a: Lunala' into a target string.
    Returns 'opp1', 'opp2', 'ally', or 'self'.
    """
    if not target_ident or target_ident.startswith('['):
        return 'self'

    target_player = _player_from_ident(target_ident)
    target_slot = _slot_from_ident(target_ident)

    if target_player == acting_player:
        return 'self' if target_slot == _slot_from_ident(target_ident) else 'ally'
    else:
        return f'opp{target_slot + 1}'


# ─── Dataset Loader ───────────────────────────────────────────────────────────

def load_battles_from_file(
    filepath: str,
    max_battles: Optional[int] = None,
    skip_errors: bool = True,
) -> List[ParsedBattle]:
    """Load and parse all battles from a JSON replay file."""
    print(f"Loading {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    battles = []
    errors = 0
    keys = list(data.keys())

    if max_battles:
        keys = keys[:max_battles]

    for i, battle_id in enumerate(keys):
        if i % 1000 == 0:
            print(f"  Parsed {i}/{len(keys)} battles, {errors} errors...")
        try:
            log = data[battle_id][1]
            battle = parse_battle_log(battle_id, log)
            if battle is not None:
                battles.append(battle)
        except Exception as e:
            errors += 1
            if not skip_errors:
                raise

    print(f"  Done. {len(battles)} valid battles, {errors} errors.")
    return battles


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    battles = load_battles_from_file(
        'data/replays/logs_gen9vgc2025regi.json',
        max_battles=100,
    )

    print(f"\nParsed {len(battles)} battles")

    if battles:
        b = battles[0]
        print(f"\nSample battle: {b.battle_id}")
        print(f"  {b.p1_name} vs {b.p2_name}")
        print(f"  Winner: {b.winner}")
        print(f"  Format: {b.format}")
        print(f"  Turns:  {len(b.turns)}")

        if b.turns:
            state, p1_act, p2_act = b.turns[0]
            print(f"\n  Turn 1 state:")
            print(f"    P1 active: {state.p1_active}")
            print(f"    P2 active: {state.p2_active}")
            print(f"    Weather:   {state.weather}")
            print(f"  Turn 1 actions:")
            print(f"    P1 slot0: {p1_act.slot0}")
            print(f"    P1 slot1: {p1_act.slot1}")
            print(f"    P2 slot0: {p2_act.slot0}")
            print(f"    P2 slot1: {p2_act.slot1}")

        # Stats across all battles
        total_turns = sum(len(b.turns) for b in battles)
        avg_turns = total_turns / len(battles)
        print(f"\nStats across {len(battles)} battles:")
        print(f"  Total turns: {total_turns}")
        print(f"  Avg turns per battle: {avg_turns:.1f}")

        # Action distribution
        move_count = switch_count = none_count = 0
        for b in battles:
            for state, p1_act, p2_act in b.turns:
                for act in [p1_act.slot0, p1_act.slot1,
                            p2_act.slot0, p2_act.slot1]:
                    if act is None:
                        none_count += 1
                    elif act[0] == 'move':
                        move_count += 1
                    elif act[0] == 'switch':
                        switch_count += 1

        total_actions = move_count + switch_count + none_count
        print(f"  Move actions:   {move_count} ({100*move_count/total_actions:.1f}%)")
        print(f"  Switch actions: {switch_count} ({100*switch_count/total_actions:.1f}%)")
        print(f"  None actions:   {none_count} ({100*none_count/total_actions:.1f}%)")