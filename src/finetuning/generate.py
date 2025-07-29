import json
import sys

sys.path.append('/root/autodl-tmp/RoG/qwen/src/')
from fileloader.dataloader import dataf

# 系统提示语
SYSTEM_PROMPT = (
    "你是视觉推理助手。请先识别图像中的对象及其属性，然后根据问题构建合理的关系路径，最后给出答案。"
)

# 图像基础路径模板
IMAGE_PATH_TEMPLATE = "/root/autodl-tmp/RoG/qwen/data/OKVQA/train2014/COCO_train2014_{:012d}.jpg"


def extract_predicted_paths(input_text):
    """从输入文本中提取Predicted Paths部分"""
    start_marker = "Predicted Paths:\n"
    end_marker = "\n\nGround Truth Paths:"

    start_index = input_text.find(start_marker) + len(start_marker)
    end_index = input_text.find(end_marker)

    if start_index == -1 or end_index == -1:
        return ""  # 如果没有找到相应的标记，则返回空字符串

    return input_text[start_index:end_index].strip()


def convert_jsonl_line(line, dataset_str,dataset,image_path):
    data = json.loads(line)

    # 提取字段
    qid = int(data["id"])
    question = dataset.getquestion(qid)
    generated_answers = data.get("prediction")
    first_answer = generated_answers[0] if generated_answers else ""

    # 提取Predicted Paths
    predicted_paths = extract_predicted_paths(data.get("input", ""))

    # 构造 messages
    assistant_content = "\n\nPredicted Paths:\n" + predicted_paths+". " + first_answer
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"<image>{question}"},
        {"role": "assistant", "content": assistant_content +" Therefore, the possible answers include: " +dataset_str},
    ]


    # 组合成单个样本结构
    return {
        "messages": messages,
        "images": [image_path]
    }


def process_jsonl_to_json(input_path, output_path, dataset):
    results = []

    with open(input_path, 'r', encoding='utf-8') as fin:
        i=0
        for idx, line in enumerate(fin):
            try:

                dataset_str =dataset.train[i].answer
                image_path = dataset.train[i].image
                converted = convert_jsonl_line(line, dataset_str,dataset, image_path)
                i+=1
                results.append(converted)

                if idx % 1000 == 0:
                    print(f"已处理 {idx} 行...")

            except Exception as e:
                print(f"处理第 {idx} 行时出错: {e}")

    # 写入最终的 JSON 文件
    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)

    print(f"✅ 转换完成！输出文件保存至: {output_path}")


# 示例调用
input_jsonl = "/root/autodl-tmp/RoG/qwen/results/multimodal/OKVQA/qwen/train/predictions.jsonl"  # 替换为你的输入文件路径
output_json = "converted_output_qwen_FVQA.json"  # 输出为 .json 文件
qapath = "/root/autodl-tmp/RoG/qwen/data/FVQA/new_dataset_release/all_qs_dict_release.json"
image = "/root/autodl-tmp/RoG/qwen/data/FVQA/new_dataset_release/images"
dataset = dataf(qapath, image)  # 加载 OKVQA
process_jsonl_to_json(input_jsonl, output_json, dataset)