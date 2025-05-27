import os
import json

dataset_dir = '../../dataset'
result = []

for user_dir in os.listdir(dataset_dir):
    user_path = os.path.join(dataset_dir, user_dir)
    if not os.path.isdir(user_path):
        continue

    history_path = os.path.join(user_path, 'history.json')
    candidate_path = os.path.join(user_path, 'candidate.json')

    # 统计history.json
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            try:
                history_items = json.load(f)
                history_count = len(history_items)
            except Exception:
                history_count = 0
    else:
        history_count = 0

    # 统计candidate.json
    if os.path.exists(candidate_path):
        with open(candidate_path, 'r') as f:
            try:
                candidate_items = json.load(f)
                candidate_count = len(candidate_items)
            except Exception:
                candidate_count = 0
    else:
        candidate_count = 0

    result.append({
        'user': user_dir,
        'history_count': history_count,
        'candidate_count': candidate_count
    })

# 输出统计结果
with open('item_num.json', 'w') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

for entry in result:
    print(f"{entry['user']}: history={entry['history_count']}, candidate={entry['candidate_count']}") 