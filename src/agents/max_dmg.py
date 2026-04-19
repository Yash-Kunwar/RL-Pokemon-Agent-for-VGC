import random
from poke_env.player import Player
from poke_env.battle.double_battle import DoubleBattle
from poke_env.player.battle_order import (
    DoubleBattleOrder,
    SingleBattleOrder,
    PassBattleOrder,
    DefaultBattleOrder,
)
from poke_env.battle.move import Move


class MaxDamagePlayer(Player):
    def choose_move(self, battle: DoubleBattle):
        orders = []
        switched_in = None

        # Handle full force switch (both slots need to switch)
        if any(battle.force_switch):
            return self.choose_random_doubles_move(battle)

        for mon, moves, switches in zip(
            battle.active_pokemon,
            battle.available_moves,
            battle.available_switches,
        ):
            # Filter out already switched-in mon
            switches = [s for s in switches if s != switched_in]

            # Empty slot
            if not mon or mon.fainted:
                orders.append(PassBattleOrder())
                continue

            # No moves but can switch
            if not moves and switches:
                mon_to_switch = random.choice(switches)
                orders.append(SingleBattleOrder(mon_to_switch))
                switched_in = mon_to_switch
                continue

            # No moves and no switches
            if not moves:
                orders.append(DefaultBattleOrder())
                continue

            # Score each move: base_power * STAB * type_effectiveness
            best_move = None
            best_score = -1

            for move in moves:
                if move.base_power == 0:
                    continue

                score = move.base_power

                # STAB bonus
                if move.type in mon.types:
                    score *= 1.5

                # Type effectiveness against opponents
                opp_multipliers = []
                for opp in battle.opponent_active_pokemon:
                    if opp and not opp.fainted:
                        opp_multipliers.append(opp.damage_multiplier(move))

                if opp_multipliers:
                    score *= max(opp_multipliers)

                if score > best_score:
                    best_score = score
                    best_move = move

            if best_move is None:
                orders.append(DefaultBattleOrder())
                continue

            # Pick target
            targets = battle.get_possible_showdown_targets(best_move, mon)
            opp_targets = [
                t for t in targets
                if t in {
                    DoubleBattle.OPPONENT_1_POSITION,
                    DoubleBattle.OPPONENT_2_POSITION,
                }
            ]
            target = random.choice(opp_targets) if opp_targets else random.choice(targets)
            orders.append(SingleBattleOrder(best_move, move_target=target))

        # Combine both slot orders
        joined = DoubleBattleOrder.join_orders([orders[0]], [orders[1]])
        if joined:
            return joined[0]
        return DoubleBattleOrder(orders[0], DefaultBattleOrder())