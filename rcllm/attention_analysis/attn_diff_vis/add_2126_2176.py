import os

# Target filename range
file_indices = range(0, 100, 10)
file_template = 'global_top_diff_positions_{}.txt'
folder = os.path.dirname(__file__)

# Number range to add
add_nums = set(str(i) for i in range(2126, 2177))

for idx in file_indices:
    file_path = os.path.join(folder, file_template.format(idx))
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, skipping.")
        continue
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    # Only add numbers that don't exist
    existing = set(lines)
    to_add = sorted(add_nums - existing, key=int)
    if to_add:
        with open(file_path, 'a', encoding='utf-8') as f:
            for num in to_add:
                f.write(f"{num}\n")
        print(f"Added to {file_path}: {to_add}")
    else:
        print(f"{file_path} already contains all 1241-1282, no need to add.")