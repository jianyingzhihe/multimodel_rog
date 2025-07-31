import json
import tqdm
from PIL import Image
from google import genai
from .src.fileloader.dataloader import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.genai import types
# 设置 API 密钥
client = genai.Client(api_key="AIzaSyD7eWUthZIzI4d9rpl07kmRb9ExDVsXp8g")

# Gemini 图像 + 文本推理函数
def gemini_vision_query(client, image_path, question):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[image, question],
            config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=-1)
                # Turn off thinking:
                # thinking_config=types.ThinkingConfig(thinking_budget=0)
                # Turn on dynamic thinking:
                # thinking_config=types.ThinkingConfig(thinking_budget=-1)
            ),
        )
        return response.text
    except Exception as e:
        print(f"Error during Gemini API call for '{question}': {e}")
        return None

# 单个样本的处理函数
def process_item(client, item):
    qid = item.id
    image_path = item.image_path
    question = item.question

    answer = gemini_vision_query(client, image_path, question)

    return {
        "id": qid,
        "question": question,
        "answer": answer
    }

# 多线程推理函数
def generate_with_gemini(client, dataset, output_path, max_workers=32):
    results = []
    lock = object()  # 用于线程安全写入的锁（虽然写入时不用锁，因为我们先收集结果）

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(process_item, client, item): item for item in dataset.combined
        }

        for future in tqdm.tqdm(as_completed(future_to_item), total=len(future_to_item), desc="Processing with Gemini"):
            result = future.result()
            if result:
                results.append(result)

    # 将结果一次性写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"✅ Results saved to {output_path}")


# 主程序入口
if __name__ == "__main__":
    qapath="/root/autodl-tmp/RoG/qwen/data/FVQA/new_dataset_release/all_qs_dict_release.json"
    image="/root/autodl-tmp/RoG/qwen/data/FVQA/new_dataset_release/images"
    ds=dataf(qapath,image)
    outputdir="temp.jsonl"
    generate_with_gemini(client, ds, outputdir)

    # 模拟评估
    # metrics = dataset.evaluate_jsonl(outputdir)
    # print("Evaluation Metrics:", metrics)