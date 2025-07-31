import json
import os
import argparse
import tqdm
from PIL import Image
import google.generativeai as genai
from dataloader import dataf
from concurrent.futures import ThreadPoolExecutor, as_completed
# 设置 API 密钥
genai.configure(api_key="your_api_key")

# 全局模型变量
model = None

# Gemini 图像 + 文本推理函数
def gemini_vision_query(image_path, question):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return ""

    # Gemini API 调用
    try:
        response = model.generate_content([image, question])
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return ""

# 单个样本的处理函数
def process_item(item):
    qid = item.id
    image_path = item.image
    question = item.question

    print(f"Processing sample ID: {qid}")
    print(f"Question: {question}")
    print(f"Image path: {image_path}")

    answer = gemini_vision_query(image_path, question)
    
    # 构建输出字典
    output_dict = {
        "id": qid,
        "question": question,
        "answer": answer
    }
    
    print(f"Answer: {answer}")
    print("=" * 50)

    return output_dict

# 多线程推理函数
def generate_with_gemini(dataset, output_path, max_workers=2):
    processed_ids = []
    # 检查输出文件是否存在，如果存在则读取已处理的ID
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                temp = json.loads(line)
                id = temp["id"]
                processed_ids.append(id)
        print(f"Found {len(processed_ids)} already processed items")
    else:
        print("Output file does not exist, starting from scratch")

    # 过滤出未处理的项目
    unprocessed_items = [item for item in dataset.val if item.id not in processed_ids]
    print(f"Processing {len(unprocessed_items)} remaining items")
    
    if len(unprocessed_items) == 0:
        print("All items have been processed!")
        return

    with open(output_path, 'a', encoding='utf-8') as f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {
                executor.submit(process_item, item): item for item in unprocessed_items
            }

            for future in tqdm.tqdm(as_completed(future_to_item), total=len(future_to_item), desc="Processing with Gemini"):
                try:
                    result = future.result()
                    if result:
                        # 实时写入结果，确保不会丢失数据
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()
                except Exception as e:
                    item = future_to_item[future]
                    print(f"Error processing item {item.id}: {e}")

    print(f"✅ Results saved to {output_path}")

def generate_with_gemini_single(dataset, output_path):
    """
    使用 Gemini API 对数据集中的单个样本进行视觉推理测试，并将结果保存为 JSONL。
    """
    processed_ids = []
    # 检查输出文件是否存在，如果存在则读取已处理的ID
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                temp = json.loads(line)
                id = temp["id"]
                processed_ids.append(id)
        print(f"Found {len(processed_ids)} already processed items")
    else:
        print("Output file does not exist, starting from scratch")
    
    with open(output_path, 'a', encoding='utf-8') as f:
        for item in dataset.combined:
            id = item.id
            if id in processed_ids:
                print(f"Skipping already processed item: {id}")
                continue
                
            result = process_item(item)
            
            # 写入 JSONL
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            
            break  # 只处理第一个样本

# 主程序入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini vision model validation script")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", 
                       choices=["gemini-1.5-flash", "gemini-2.5-pro", "gemini-2.5-flash"],
                       help="Gemini model to use")
    parser.add_argument("--dataset_type", type=str, default="fvqa", 
                       choices=["fvqa"],
                       help="Type of dataset to use")
    parser.add_argument("--qapath", type=str, default="new_dataset_release/new_dataset_release/all_qs_dict_release.json",
                       help="Path to QA data file")
    parser.add_argument("--image_dir", type=str, default="new_dataset_release/new_dataset_release/images",
                       help="Path to images directory")
    parser.add_argument("--test_single", action="store_true", 
                       help="Run single sample test instead of full processing")
    parser.add_argument("--max_workers", type=int, default=16,
                       help="Maximum number of worker threads for parallel processing")
    
    args = parser.parse_args()
    
    # 初始化模型
    model = genai.GenerativeModel(args.model)
    print(f"🤖 Initialized model: {args.model}")
    
    # 加载数据集
    dataset = dataf(args.qapath, args.image_dir)
    print(f"📊 Loaded dataset: {args.dataset_type}")
    
    # 生成输出文件名 (基于模型和数据集名称)
    model_name_clean = args.model.replace(".", "_").replace("-", "_")
    
    if args.test_single:
        # 单样本测试
        outputdir = f"single_test_result_{model_name_clean}_{args.dataset_type}.jsonl"
        print(f"🧪 Running single sample test with {args.model}...")
        generate_with_gemini_single(dataset, outputdir)
        print(f"✅ Single sample test result saved to {outputdir}")
    else:
        # 全量并行处理
        outputdir = f"output_{model_name_clean}_{args.dataset_type}_parallel.jsonl"
        print(f"🚀 Running full dataset processing with {args.model}...")
        print(f"📈 Using {args.max_workers} worker threads")
        generate_with_gemini(dataset, outputdir, max_workers=args.max_workers)
        
        # 可选评估
        print(f"\n📋 To evaluate results, you can run:")
        print(f"dataset.evaluate_jsonl('{outputdir}')")
