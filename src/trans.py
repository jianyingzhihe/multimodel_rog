from ast import literal_eval
import json

input_path = "../results/gen_rule_path/OKVQA/Qwen2.5-VL-7B-Instruct/val/predictions_3_False_feed.jsonl"
output_path = "../results/output.json"

with open(input_path, "r") as fin:
    data = [literal_eval(line.strip()) for line in fin]

with open(output_path, "w", encoding="utf-8") as fout:
    json.dump(data, fout, ensure_ascii=False, indent=4)

print("转换完成！")