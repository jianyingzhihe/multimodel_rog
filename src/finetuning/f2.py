import json
import os

# 参数配置
input_json_path = "/root/autodl-tmp/RoG/qwen/src/finetuning/train_relation_path_full.json"
image_root = "/root/autodl-tmp/RoG/qwen/data/OKVQA/train2014"
output_json_path = "/root/autodl-tmp/RoG/qwen/src/finetuning/converted_dataset_for_relation_path_generation.json"

# 读取原始数据
with open(input_json_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

converted_data = []

for item in raw_data:
    image_path = item.get("image_path")
    question = item.get("question")
    relation_paths = item.get("relation_paths")
    answers = item.get("answers")

    # 检查必要字段是否存在
    if not all([image_path, question, relation_paths, answers]):
        print(f"警告: 数据项缺少字段，跳过: {item}")
        continue

    full_image_path = os.path.join(image_root, image_path)
    if not os.path.exists(full_image_path):
        print(f"警告: 图像文件不存在，跳过: {full_image_path}")
        continue

    # 构造用户输入内容
    user_content = f"<image>{question}"

    # 构造助手输出内容：思考过程 + 关系路径 + 答案
    answer_str = "\n".join([f"答案{i+1}: {ans}" for i, ans in enumerate(answers) if ans])
    relation_path_str = "\n".join([f"{i+1}. {p}" for i, p in enumerate(relation_paths)])

    assistant_content = (
        "好的，我将从图像中提取相关信息，并基于问题构建关系路径。\n\n"
        f"Relation Paths:\n{relation_path_str}\n\n"
        f"{answer_str}"
    )

    # 构建 messages 对话结构
    messages = [
        {
            "role": "system",
            "content": "你是视觉推理助手。请先识别图像中的对象及其属性，然后根据问题构建合理的关系路径，最后给出答案。"
        },
        {
            "role": "user",
            "content": user_content
        },
        {
            "role": "assistant",
            "content": assistant_content
        }
    ]

    converted_item = {
        "messages": messages,
        "images": [full_image_path]
    }

    converted_data.append(converted_item)

# 保存为 JSON 文件
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=2)

print(f"✅ 数据已成功转换并保存至: {output_json_path}")