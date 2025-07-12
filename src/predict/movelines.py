import os
from fileloader import dataloader

def extract_last_n_lines(input_file, output_file, n=5046):
    # 获取文件总行数
    with open(input_file, 'r', encoding='utf-8') as f:
        line_count = sum(1 for line in f)

    print(f"Total lines in input file: {line_count}")

    # 计算从哪一行开始读取最后 n 行
    start_line = max(0, line_count - n)

    print(f"Extracting last {n} lines (starting from line {start_line})")

    # 再次打开文件，逐行读取最后 n 行
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for i, line in enumerate(fin):
            if i >= start_line:
                fout.write(line)

    print(f"Done. Last {n} lines written to '{output_file}'.")


if __name__ == "__main__":
    # input_jsonl = "/root/autodl-tmp/RoG/qwen/results/multimodal/OKVQA/Qwen2.5-VL-7B-Instruct/val/predictions.jsonl"  # 替换为你的输入文件路径
    output_jsonl = "/root/autodl-tmp/RoG/qwen/results/multimodal/OKVQA/gemma/val/predictions.jsonl"  # 输出文件路径
    #
    # extract_last_n_lines(input_jsonl, output_jsonl, n=5046)
    a=dataloader.datas("/root/autodl-tmp/RoG/qwen/data/OKVQA")
    a.evaluate_jsonl(output_jsonl)
