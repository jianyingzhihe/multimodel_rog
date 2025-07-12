import json
import os

# 原始数据文件路径
input_jsonl = "/root/autodl-tmp/RoG/qwen/results/multimodal/OKVQA/Qwen2.5-VL-7B-Instruct/train/predictions_parsed.jsonl"
output_json = "train_relation_path_full.json"
image_root = "../../data/OKVQA/train2014"  # 这里只是用于说明，实际在训练时使用

processed_data = []

with open(input_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line.strip())

        qid = item["id"]
        question = item["question"]

        # 构造图像路径（不包含根目录，训练时再拼接）
        image_path = f"COCO_train2014_{qid:012d}.jpg"

        # 获取所有 ground truth relation paths（保留原始列表）
        relation_paths = item["input"]["ground_truth_paths"]

        # 获取所有 ground truth answers（保留原始列表，不做处理或按需处理）
        answers = item["ground_truth"]

        processed_data.append({
            "image_path": image_path,
            "question": question,
            "relation_paths": relation_paths,
            "answers": answers
        })

# 保存成标准的训练用 JSON 文件
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, indent=2, ensure_ascii=False)

print(f"已生成 {len(processed_data)} 条训练样本，保存至 {output_json}")