import asyncio
import sys
sys.path.insert(0, '.')

import torch
from poke_env import LocalhostServerConfiguration
from poke_env.player import RandomPlayer
from src.agents.transformer_player import TransformerPlayer
from src.models.transformer_policy import VGCPolicyNetwork


async def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VGCPolicyNetwork()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    transformer_player = TransformerPlayer(
        model=model,
        device=device,
        mode="greedy",
        battle_format="gen9randomdoublesbattle",
        server_configuration=LocalhostServerConfiguration,
        log_level=25,
    )
    random_player = RandomPlayer(
        battle_format="gen9randomdoublesbattle",
        server_configuration=LocalhostServerConfiguration,
        log_level=25,
    )

    n_battles = 5
    await transformer_player.battle_against(random_player, n_battles=n_battles)

    print(f"\n=== Results (untrained model vs random) ===")
    print(f"Transformer wins: {transformer_player.n_won_battles} / {n_battles}")
    print(f"Random wins:      {random_player.n_won_battles} / {n_battles}")
    print(f"Turns recorded:   {len(transformer_player.turn_data)}")

    if transformer_player.turn_data:
        last = transformer_player.turn_data[-1]
        print(f"\nLast turn sample:")
        print(f"  Turn:     {last['turn']}")
        print(f"  Action 0: {last['action_0']}")
        print(f"  Action 1: {last['action_1']}")
        print(f"  Value est:{last['value']:.4f}")
        print(f"  Obs mean: {last['obs'].mean():.4f}")
        print(f"  Obs std:  {last['obs'].std():.4f}")


asyncio.run(main())