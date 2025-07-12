import sys
import os
import json
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from tqdm import tqdm
from fileloader.llama import *  # 假设 dataloader.py 在当前目录下
import argparse
# # 导入 eval_result 函数
# from evaluate_results import eval_result  # 确保路径正确


def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["id"])
        fout = open(path, "a")
        return fout, processed_results

def format_paths(paths, title="Predicted Paths"):
    if not paths:
        return ""
    path_str = "\n".join([f"Path {i+1}: {' -> '.join(p)}" for i, p in enumerate(paths)])
    return f"{title}:\n{path_str}\n\n"

def main(args):
    print("Loading dataset...")
    dataset = datas(args.data_dir, type=args.split)  # 加载 OKVQA
    print(args.model_path)
    print(os.getcwd())
    model = llamamod(modelpath=args.model_path,type="hf")

    output_dir = os.path.join(
        args.predict_path,
        args.d,
        args.model_name,
        args.split,
    )
    print("Save results to:", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "predictions.jsonl")
    fout, processed_list = get_output_file(output_file, force=args.force)

    total = len(dataset.combined)
    correct = 0
    generated_rule_path=[]
    with open(args.rule_path, "r") as f:
        for line in f:
            data=json.loads(line.strip())
            generated_rule_path.append(data)
    if(len(generated_rule_path)==total):
        i=0
        for item in tqdm(dataset.combined, total=total):
            image_id = item["id"]
            raw_question = item["question"]  # 原始问题
            image_path = item["image_path"]
            ground_truth = [ans["answer"] for ans in item["answer"]]
            print(image_id)
            # 获取预测路径 & 真实路径
            rule_data = generated_rule_path[i]
            i+=1
            ground_path = rule_data["ground_paths"]
            prediction_path = rule_data["prediction"]

            try:
                path_prompt = ""
                path_prompt += format_paths(prediction_path, "Predicted Paths")
                path_prompt += format_paths(ground_path, "Ground Truth Paths")

                enhanced_question = path_prompt + raw_question

                predictions = model.inf_with_score(enhanced_question, image_path, num_beams=args.beam_size)
                prediction_texts = [pred['answer'] for pred in predictions]
            except Exception as e:
                print(f"Error processing {image_id}: {e}")
                continue

            result = {
                "id": image_id,
                "question": raw_question,
                "prediction": prediction_texts,
                "input": enhanced_question,  # 使用增强后的输入
            }

            fout.write(json.dumps(result) + "\n")
            fout.flush()

    else :
        print("length is not same")

    fout.close()
    print("Prediction finished.")

    # 调用评估函数
    # eval_result(output_file, cal_f1=True, topk=args.top_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-modal QA prediction with Qwen2.5-VL")

    parser.add_argument("--data_dir", type=str, default="../data/OKVQA", help="Path to the OKVQA dataset")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Dataset split (train or val)")
    parser.add_argument("--predict_path", type=str, default="results/multimodal", help="Directory to save predictions")
    parser.add_argument("--d", "-d", type=str, default="OKVQA", help="Dataset name")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-7B-Instruct", help="Model name for saving results")
    parser.add_argument("--model_path", type=str, default="/root/models/Qwen2.5-VL-7B-Instruct", help="Local path of the model weights")
    parser.add_argument("--beam_size", type=int, default=3, help="Number of beam search candidates")
    parser.add_argument("--force", "-f", action="store_true", help="Force overwrite existing results")
    parser.add_argument("--top_k", type=int, default=-1, help="Top K predictions to consider for evaluation")
    parser.add_argument('--cal_f1', action="store_true", help="Calculate F1 score")
    parser.add_argument("--rule_path",type=str,default=f"./results/gen_rule_path/",help="Path to the generated rules")
    parser.add_argument("--engine_type",type=str,default="hf")
    args = parser.parse_args()

    main(args)