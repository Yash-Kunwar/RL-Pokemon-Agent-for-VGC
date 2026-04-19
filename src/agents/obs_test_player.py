import numpy as np
from poke_env.player import Player
from poke_env.battle.double_battle import DoubleBattle
from poke_env.player.battle_order import BattleOrder
from src.utils.observation import embed_battle, get_observation_size


class ObsTestPlayer(Player):
    """
    Plays randomly but calls embed_battle every turn and validates the output.
    This verifies the observation space works correctly on real battle states.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_sizes = []
        self.errors = []

    def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        try:
            obs = embed_battle(battle)

            # Validate shape
            expected = get_observation_size()
            assert obs.shape == (expected,), f"Shape mismatch: {obs.shape} vs ({expected},)"

            # Validate no NaN or Inf values
            assert not np.isnan(obs).any(), "NaN found in observation"
            assert not np.isinf(obs).any(), "Inf found in observation"

            # Validate range — most values should be in [-1, 1]
            out_of_range = np.sum((obs < -1.0) | (obs > 1.0))
            if out_of_range > 0:
                self.errors.append(f"Turn {battle.turn}: {out_of_range} values out of [-1,1]")

            self.obs_sizes.append(obs.shape[0])

        except Exception as e:
            self.errors.append(f"Turn {battle.turn}: {str(e)}")

        return self.choose_random_move(battle)