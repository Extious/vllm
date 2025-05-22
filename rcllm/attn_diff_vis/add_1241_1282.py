import os

# 目标文件名范围
file_indices = range(0, 100, 10)
file_template = 'global_top_diff_positions_{}.txt'
folder = os.path.dirname(__file__)

# 要添加的数字区间
add_nums = set(str(i) for i in range(1241, 1283))

for idx in file_indices:
    file_path = os.path.join(folder, file_template.format(idx))
    if not os.path.exists(file_path):
        print(f"{file_path} 不存在，跳过。")
        continue
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    # 只添加不存在的数字
    existing = set(lines)
    to_add = sorted(add_nums - existing, key=int)
    if to_add:
        with open(file_path, 'a', encoding='utf-8') as f:
            for num in to_add:
                f.write(f"{num}\n")
        print(f"已向 {file_path} 添加: {to_add}")
    else:
        print(f"{file_path} 已包含全部1241-1282，无需添加。") 