import sys
sys.path.insert(0, '.')
from src.data.replay_parser import load_battles_from_file

total_battles = 0
total_turns = 0

files = [
    'data/replays/logs_gen9vgc2025regi.json',
    'data/replays/logs_gen9vgc2025regh.json',
    'data/replays/logs_gen9vgc2024regh.json',
    'data/replays/logs_gen9vgc2024regg.json',
]

for f in files:
    battles = load_battles_from_file(f)
    turns = sum(len(b.turns) for b in battles)
    total_battles += len(battles)
    total_turns += turns
    fname = f.split('/')[-1]
    print(f'  {fname}: {len(battles)} battles, {turns} turns')

print(f'\nTotal: {total_battles} battles, {total_turns} turns')
print(f'Estimated training samples (x2 players): {total_battles * 2}')
print(f'Estimated total turns (x2 players): {total_turns * 2}')