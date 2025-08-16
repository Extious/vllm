import os

global_diff_path = 'global_top_diff_positions.txt'
input_path = 'input_prompt_token_positions.txt'
output_path = 'input_prompt_token_positions_marked.txt'

def main():
    # Read token positions that need to be marked
    with open(global_diff_path, 'r') as f:
        diff_tokens = set()
        for line in f:
            line = line.strip()
            if line.isdigit():
                diff_tokens.add(int(line))

    # Read original file and mark
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            line_strip = line.rstrip('\n')
            # Get the first token position number of each line
            parts = line_strip.split('\t')
            if parts and parts[0].isdigit() and int(parts[0]) in diff_tokens:
                fout.write(f'{line_strip} #DIFF\n')
            else:
                fout.write(f'{line_strip}\n')

if __name__ == '__main__':
    main() 