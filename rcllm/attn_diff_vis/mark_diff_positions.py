import os

global_diff_path = 'global_top_diff_positions.txt'
input_path = 'input_prompt_token_positions.txt'
output_path = 'input_prompt_token_positions_marked.txt'

def main():
    # 读取需要标记的token位置
    with open(global_diff_path, 'r') as f:
        diff_tokens = set()
        for line in f:
            line = line.strip()
            if line.isdigit():
                diff_tokens.add(int(line))

    # 读取原始文件并标记
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            line_strip = line.rstrip('\n')
            # 获取每行第一个token位置编号
            parts = line_strip.split('\t')
            if parts and parts[0].isdigit() and int(parts[0]) in diff_tokens:
                fout.write(f'{line_strip} #DIFF\n')
            else:
                fout.write(f'{line_strip}\n')

if __name__ == '__main__':
    main() 