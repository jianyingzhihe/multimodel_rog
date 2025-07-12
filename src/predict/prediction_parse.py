import json


def parse_input_field(input_str):
    lines = input_str.strip().split('\n')

    result = {
        "predicted_paths": [],
        "ground_truth_paths": []
    }

    current_section = None  # 可以是 'predicted' 或 'ground_truth'

    for line in lines:
        line = line.strip()
        if line.startswith("Predicted Paths:"):
            current_section = "predicted"
        elif line.startswith("Ground Truth Paths:"):
            current_section = "ground_truth"
        elif line.startswith("Path"):
            path_part = line.split(":", 1)[1].strip()
            nodes = [x.strip() for x in path_part.split("->")]

            if current_section == "predicted":
                result["predicted_paths"].append(nodes)
            elif current_section == "ground_truth":
                result["ground_truth_paths"].append(nodes)

    return result

def join_path_list(path_list):
    return ''.join(path_list)


if __name__ == '__main__':
    prediction_path = "../../results/multimodal/OKVQA/llama/train/predictions.jsonl"
    output_path = "../../results/multimodal/OKVQA/Qwen2.5-VL-7B-Instruct/train/predictions_parsed.jsonl"

    predictions = []

    with open(prediction_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    predictions.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e} in line: {line[:50]}...")

    # 解析 input 字段并转换 ground_truth_paths
    for item in predictions:
        raw_input = item['input']
        parsed_input = parse_input_field(raw_input)
        parsed_input['ground_truth_paths'] = [join_path_list(p) for p in parsed_input['ground_truth_paths']]
        item['input_parsed'] = parsed_input  # 使用新字段保存结构化数据
        # 如果你想替换掉旧的 input 字段：
        del item['input']  # 可选：删除原始 input 字段
        item['input'] = parsed_input

    # 写入新文件
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in predictions:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"已将处理后的数据保存至：{output_path}")

