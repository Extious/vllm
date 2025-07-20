import matplotlib.pyplot as plt
import os
from collections import Counter, defaultdict

# Filenames and corresponding top-k
files = [
    ("global_top_diff_positions_10.txt", 10),
    ("global_top_diff_positions_20.txt", 20),
    ("global_top_diff_positions_30.txt", 30),
    ("global_top_diff_positions_40.txt", 40),
    ("global_top_diff_positions_50.txt", 50),
    ("global_top_diff_positions_60.txt", 60),
    ("global_top_diff_positions_70.txt", 70),
    ("global_top_diff_positions_80.txt", 80),
    ("global_top_diff_positions_90.txt", 90),
]

position_in_file_count = defaultdict(int)
all_positions = set()

# Count how many files each position appears in
for fname, _ in files:
    path = os.path.join(os.path.dirname(__file__), fname)
    with open(path) as f:
        positions = set(int(line.strip()) for line in f if line.strip())
    for pos in positions:
        position_in_file_count[pos] += 1
    all_positions.update(positions)

all_positions = sorted(all_positions)

# Divide into six parts
n = len(all_positions)
parts = [
    (i * n // 6, (i + 1) * n // 6) for i in range(6)
]

for i, (start, end) in enumerate(parts, 1):
    plt.figure(figsize=(20, 10))
    part_positions = all_positions[start:end]
    # Only keep positions with data in this range
    valid_positions = [pos for pos in part_positions if position_in_file_count[pos] > 0]
    if not valid_positions:
        continue
    y = [position_in_file_count[pos] / 9 for pos in valid_positions]
    plt.plot(valid_positions, y, marker='o')
    plt.xlabel("Position")
    plt.ylabel("Probability")
    plt.title(f"Top Diff Positions Probability (Part {i})")
    plt.ylim(0, 1)
    plt.xticks(valid_positions, rotation=90)
    plt.tight_layout()
    plt.savefig(f"top_diff_positions_part{i}.png")
    plt.close() 