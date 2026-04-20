import numpy as np
from typing import Dict, Optional
from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.effect import Effect
from poke_env.battle.status import Status
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.field import Field


# ─── Reward Constants ─────────────────────────────────────────────────────────

REWARDS = {
    # Terminal
    'win':              +1.00,
    'loss':             -1.00,
    'mon_alive_end':    +0.02,

    # Faints
    'faint_opponent':   +0.10,
    'faint_ally':       -0.10,

    # Damage
    'damage_dealt':     +0.05,
    'damage_taken':     -0.05,
    'se_hit':           +0.05,

    # Status
    'status_inflicted': +0.08,
    'status_received':  -0.08,

    # Trapping
    'trap_opponent':    +0.03,
    'trapped':          -0.03,

    # Field control
    'tailwind_set':     +0.04,
    'tailwind_opp':     -0.04,
    'screen_set':       +0.03,

    # Entry hazards
    'hazard_set':       +0.03,   
    'hazard_opp':       -0.03,   

    # Trick Room
    'trickroom_set':    +0.04,
    'trickroom_opp':    -0.04,

    # Stat changes
    'boost_gained':     +0.06,
    'boost_lost':       -0.06,
}

# Effects that count as trapping
TRAP_EFFECTS = {
    Effect.LEECH_SEED,
    Effect.ENCORE,
    Effect.BIND,
    Effect.WRAP,
    Effect.CLAMP,
    Effect.FIRE_SPIN,
    Effect.WHIRLPOOL,
    Effect.MAGMA_STORM,
    Effect.SAND_TOMB,
    Effect.SNAP_TRAP,
    Effect.THUNDER_CAGE,
    Effect.INFESTATION,
    Effect.SALT_CURE,
    Effect.PARTIALLY_TRAPPED,
    Effect.OCTOLOCK,
    Effect.NO_RETREAT,
}

# Status conditions that matter
BAD_STATUSES = {Status.BRN, Status.PAR, Status.SLP, Status.FRZ, Status.TOX, Status.PSN}

# Screen side conditions
SCREEN_CONDITIONS = {
    SideCondition.REFLECT,
    SideCondition.LIGHT_SCREEN,
    SideCondition.AURORA_VEIL,
}

HAZARD_CONDITIONS = {
    SideCondition.SPIKES,
    SideCondition.STEALTH_ROCK,
    SideCondition.TOXIC_SPIKES,
    SideCondition.STICKY_WEB,
}

# Boost stats we track
TRACKED_BOOST_STATS = ['atk', 'def', 'spa', 'spd', 'spe']


class BattleState:
    """
    Snapshot of a battle state for reward computation.
    Stores the previous state so we can compute deltas.
    """

    def __init__(self, battle: DoubleBattle):
        self.update(battle)

    def update(self, battle: DoubleBattle):
        """Update snapshot from current battle state."""
        # HP fractions for all active Pokémon
        self.ally_hp = {
            mon.species: mon.current_hp_fraction
            for mon in battle.active_pokemon
            if mon is not None and not mon.fainted
        }
        self.opp_hp = {
            mon.species: mon.current_hp_fraction
            for mon in battle.opponent_active_pokemon
            if mon is not None and not mon.fainted
        }

        # Fainted counts
        self.ally_fainted = sum(
            1 for mon in battle.team.values() if mon.fainted
        )
        self.opp_fainted = sum(
            1 for mon in battle.opponent_team.values() if mon.fainted
        )

        # Status conditions on active Pokémon
        self.ally_statuses = {
            mon.species: mon.status
            for mon in battle.active_pokemon
            if mon is not None
        }
        self.opp_statuses = {
            mon.species: mon.status
            for mon in battle.opponent_active_pokemon
            if mon is not None
        }

        # Trapping effects on active Pokémon
        self.ally_trapped = {
            mon.species: bool(TRAP_EFFECTS.intersection(mon.effects.keys()))
            for mon in battle.active_pokemon
            if mon is not None
        }
        self.opp_trapped = {
            mon.species: bool(TRAP_EFFECTS.intersection(mon.effects.keys()))
            for mon in battle.opponent_active_pokemon
            if mon is not None
        }

        # Stat boosts
        self.ally_boosts = {
            mon.species: {stat: mon.boosts.get(stat, 0) for stat in TRACKED_BOOST_STATS}
            for mon in battle.active_pokemon
            if mon is not None
        }
        self.opp_boosts = {
            mon.species: {stat: mon.boosts.get(stat, 0) for stat in TRACKED_BOOST_STATS}
            for mon in battle.opponent_active_pokemon
            if mon is not None
        }

        # Side conditions
        self.ally_conditions = set(battle.side_conditions.keys())
        self.opp_conditions = set(battle.opponent_side_conditions.keys())

        # Fields
        self.fields = set(battle.fields.keys())

        # Turn number
        self.turn = battle.turn

        # Alive count
        self.ally_alive = sum(
            1 for mon in battle.team.values() if not mon.fainted
        )


class RewardShaper:
    """
    Computes shaped rewards by comparing consecutive battle states.
    Tracks the previous state internally.
    """

    def __init__(self, dense_weight: float = 1.0):
        """
        Args:
            dense_weight: multiplier for all dense rewards (0.0 to 1.0).
                          Decay this over training to shift focus to win/loss.
        """
        self.dense_weight = dense_weight
        self._prev_state: Optional[BattleState] = None
        self._reward_log: Dict[str, float] = {}

    def reset(self):
        """Call at the start of each new battle."""
        self._prev_state = None
        self._reward_log = {}

    def compute_reward(
        self,
        battle: DoubleBattle,
        prev_state: Optional[BattleState] = None,
    ) -> float:
        """
        Compute reward for the current turn by comparing to previous state.

        Args:
            battle:     current battle state
            prev_state: previous BattleState snapshot (if None uses internal)

        Returns:
            scalar reward float
        """
        curr_state = BattleState(battle)

        if prev_state is None:
            prev_state = self._prev_state

        # First turn — no previous state to compare
        if prev_state is None:
            self._prev_state = curr_state
            return 0.0

        reward = 0.0
        self._reward_log = {}

        # ── Terminal rewards ──────────────────────────────────────────────────
        if battle.won:
            r = REWARDS['win']
            # Bonus for each ally still alive
            alive_bonus = curr_state.ally_alive * REWARDS['mon_alive_end']
            reward += r + alive_bonus
            self._reward_log['win'] = r
            self._reward_log['mon_alive_end'] = alive_bonus
            self._prev_state = curr_state
            return reward

        if battle.lost:
            reward += REWARDS['loss']
            self._reward_log['loss'] = REWARDS['loss']
            self._prev_state = curr_state
            return reward

        # ── Dense rewards (scaled by dense_weight) ────────────────────────────
        w = self.dense_weight

        # Faints
        new_opp_fainted = curr_state.opp_fainted - prev_state.opp_fainted
        new_ally_fainted = curr_state.ally_fainted - prev_state.ally_fainted

        if new_opp_fainted > 0:
            r = new_opp_fainted * REWARDS['faint_opponent'] * w
            reward += r
            self._reward_log['faint_opponent'] = r

        if new_ally_fainted > 0:
            r = new_ally_fainted * REWARDS['faint_ally'] * w
            reward += r
            self._reward_log['faint_ally'] = r

        # Damage dealt — check HP drops on opponent active Pokémon
        for species, prev_hp in prev_state.opp_hp.items():
            if species in curr_state.opp_hp:
                hp_drop = prev_hp - curr_state.opp_hp[species]
                if hp_drop > 0.30:
                    r = REWARDS['damage_dealt'] * w
                    reward += r
                    self._reward_log['damage_dealt'] = \
                        self._reward_log.get('damage_dealt', 0) + r

        # Damage taken — check HP drops on ally active Pokémon
        for species, prev_hp in prev_state.ally_hp.items():
            if species in curr_state.ally_hp:
                hp_drop = prev_hp - curr_state.ally_hp[species]
                if hp_drop > 0.30:
                    r = REWARDS['damage_taken'] * w
                    reward += r
                    self._reward_log['damage_taken'] = \
                        self._reward_log.get('damage_taken', 0) + r

        # Status inflicted on opponents
        for species, curr_status in curr_state.opp_statuses.items():
            prev_status = prev_state.opp_statuses.get(species)
            if curr_status in BAD_STATUSES and prev_status not in BAD_STATUSES:
                r = REWARDS['status_inflicted'] * w
                reward += r
                self._reward_log['status_inflicted'] = r

        # Status received by allies
        for species, curr_status in curr_state.ally_statuses.items():
            prev_status = prev_state.ally_statuses.get(species)
            if curr_status in BAD_STATUSES and prev_status not in BAD_STATUSES:
                r = REWARDS['status_received'] * w
                reward += r
                self._reward_log['status_received'] = r

        # Trapping opponent
        for species, is_trapped in curr_state.opp_trapped.items():
            was_trapped = prev_state.opp_trapped.get(species, False)
            if is_trapped and not was_trapped:
                r = REWARDS['trap_opponent'] * w
                reward += r
                self._reward_log['trap_opponent'] = r

        # Getting trapped
        for species, is_trapped in curr_state.ally_trapped.items():
            was_trapped = prev_state.ally_trapped.get(species, False)
            if is_trapped and not was_trapped:
                r = REWARDS['trapped'] * w
                reward += r
                self._reward_log['trapped'] = r

        # Tailwind set by us
        if SideCondition.TAILWIND in curr_state.ally_conditions and \
           SideCondition.TAILWIND not in prev_state.ally_conditions:
            r = REWARDS['tailwind_set'] * w
            reward += r
            self._reward_log['tailwind_set'] = r

        # Tailwind set by opponent
        if SideCondition.TAILWIND in curr_state.opp_conditions and \
           SideCondition.TAILWIND not in prev_state.opp_conditions:
            r = REWARDS['tailwind_opp'] * w
            reward += r
            self._reward_log['tailwind_opp'] = r

        # Screens set by us
        new_screens = (curr_state.ally_conditions - prev_state.ally_conditions) \
                      .intersection(SCREEN_CONDITIONS)
        if new_screens:
            r = len(new_screens) * REWARDS['screen_set'] * w
            reward += r
            self._reward_log['screen_set'] = r

        # Entry hazards set by us
        new_hazards = (curr_state.ally_conditions - prev_state.ally_conditions) \
                      .intersection(HAZARD_CONDITIONS)
        if new_hazards:
            r = len(new_hazards) * REWARDS['hazard_set'] * w
            reward += r
            self._reward_log['hazard_set'] = r

        # Entry hazards set by opponent
        new_opp_hazards = (curr_state.opp_conditions - prev_state.opp_conditions) \
                          .intersection(HAZARD_CONDITIONS)
        if new_opp_hazards:
            r = len(new_opp_hazards) * REWARDS['hazard_opp'] * w
            reward += r
            self._reward_log['hazard_opp'] = r

        # Trick Room
        if Field.TRICK_ROOM in curr_state.fields and \
           Field.TRICK_ROOM not in prev_state.fields:
            # Determine who set it — check whose side conditions changed
            # We approximate: if our ally_conditions changed this turn, we set it
            if curr_state.ally_conditions != prev_state.ally_conditions:
                r = REWARDS['trickroom_set'] * w
                reward += r
                self._reward_log['trickroom_set'] = r
            else:
                r = REWARDS['trickroom_opp'] * w
                reward += r
                self._reward_log['trickroom_opp'] = r

        # Stat boosts gained by allies
        for species, curr_boosts in curr_state.ally_boosts.items():
            prev_boosts = prev_state.ally_boosts.get(species, {})
            for stat, curr_val in curr_boosts.items():
                prev_val = prev_boosts.get(stat, 0)
                stages_gained = curr_val - prev_val
                if stages_gained > 0:
                    r = stages_gained * REWARDS['boost_gained'] * w
                    reward += r
                    self._reward_log['boost_gained'] = \
                        self._reward_log.get('boost_gained', 0) + r

        # Stat boosts gained by opponents (bad for us)
        for species, curr_boosts in curr_state.opp_boosts.items():
            prev_boosts = prev_state.opp_boosts.get(species, {})
            for stat, curr_val in curr_boosts.items():
                prev_val = prev_boosts.get(stat, 0)
                stages_gained = curr_val - prev_val
                if stages_gained > 0:
                    r = stages_gained * REWARDS['boost_lost'] * w
                    reward += r
                    self._reward_log['boost_lost'] = \
                        self._reward_log.get('boost_lost', 0) + r

        self._prev_state = curr_state
        return reward

    def get_reward_breakdown(self) -> Dict[str, float]:
        """Returns breakdown of last reward computation."""
        return dict(self._reward_log)


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("RewardShaper initialized")
    print(f"Total reward components: {len(REWARDS)}")
    print(f"Max possible turn reward: {sum(v for v in REWARDS.values() if v > 0):.2f}")
    print(f"Min possible turn reward: {sum(v for v in REWARDS.values() if v < 0):.2f}")
    print()
    print("Reward breakdown:")
    for key, val in REWARDS.items():
        sign = '+' if val > 0 else ''
        print(f"  {key:20s}: {sign}{val:.2f}")