import json


def process_assistant_content(content):
    lines = content.split('\n')
    relation_paths_started = False
    new_lines = []
    answers_seen = set()

    for line in lines:
        # 处理 Relation Paths 行
        if line.startswith("Relation Paths:"):
            relation_paths_started = True
            new_lines.append(line)
        elif relation_paths_started and '.' in line:
            # 将 . 替换为 ->
            new_line = line.replace('.', '->', 10)
            new_lines.append(new_line)
        elif line.startswith("答案"):
            # 提取答案内容
            parts = line.split(':', 1)
            if len(parts) < 2:
                continue
            value = parts[1].strip()
            if value not in answers_seen:
                answers_seen.add(value)
        else:
            # 其他普通行直接保留
            new_lines.append(line)

    # 添加“可能的答案包括”
    if answers_seen:
        unique_answers = ', '.join(sorted(answers_seen))
        new_lines.append(f"可能的答案包括: {unique_answers}")

    return '\n'.join(new_lines)


def process_data(data):
    for item in data:
        messages = item.get("messages", [])
        for msg in messages:
            if msg["role"] == "assistant":
                msg["content"] = process_assistant_content(msg["content"])
    return data


# —————————————— 主程序入口 —————————————— #

if __name__ == "__main__":
    input_path = "converted_dataset_for_relation_path_generation.json"  # 输入文件路径
    output_path = "processed_output_qwen.json"  # 输出文件路径

    # 读取输入文件
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {input_path}")
        exit(1)
    except json.JSONDecodeError:
        print(f"❌ 错误：{input_path} 不是有效的 JSON 格式")
        exit(1)

    # 处理数据
    processed_data = process_data(raw_data)

    # 写入输出文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 已成功处理并保存到 {output_path}")