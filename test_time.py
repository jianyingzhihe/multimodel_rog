import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import argparse
from src.fileloader.dataloader import *
from src.fileloader.llama import *
from src.fileloader.qwen import *
from src.fileloader.google import *

def generate(model, dataset):

    cnt=0
    timelist=[]
    t0=time.time()
    for each in tqdm.tqdm(dataset.combined, desc="Processing images"):
        cnt+=1
        image = each.image
        question = each.question
        messages = [
                            {"role": "system",
                             "content": [
                                 {"type": "text",
                                  "text": "You are a visual reasoning assistant. Please first identify the objects and their attributes in the image, then construct a reasonable relationship path based on the question, and finally provide the answer in English."},
                             ]},
                            {"role": "user",
                             "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}
                        ]
        result = model.inf_with_messages(messages)
        total_time = time.time() - t0
        average_time = total_time / cnt
        timelist.append({"iteration": cnt, "total_time": total_time, "average_time": average_time})
        if cnt==500:
            return timelist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal model validation script")
    parser.add_argument("--dataset", type=str, default="fvqa",
                        choices=["fvqa", "aokvqa", "okvqa"],
                        help="Type of dataset to use")
    parser.add_argument("--model", type=str, default="qwen",
                        choices=["qwen", "gemma", "google", "llama", "intern"],
                        help="Type of model to use")
    args = parser.parse_args()
    # 加载不同数据集
    image_path = None
    if args.dataset == "fvqa":
        qapath = "/root/autodl-tmp/RoG/qwen/data/FVQA/new_dataset_release/all_qs_dict_release.json"
        image_path = "/root/autodl-tmp/RoG/qwen/data/FVQA/new_dataset_release/images"
        ds = dataf(qapath, image_path)
    if args.dataset == "aokvqa":
        image_path = "/root/autodl-tmp/RoG/qwen/data/AOKVQA/val2014"  # AOKVQA 图像路径
        ds = datap("/root/autodl-tmp/RoG/qwen/data/AOKVQA/data/test-00000-of-00001-d306bf3ad53b6618.parquet")
    if args.dataset == "okvqa":
        ds = datas("/root/autodl-tmp/RoG/qwen/data/OKVQA",split="val")

    if args.model == "qwen":
        model = qwenmod(modelpath="/root/autodl-tmp/RoG/qwen/multimodels/Qwen/qwenvl" )
    elif args.model == "gemma" or args.model == "google":
        model = googlemod(modelpath="/root/autodl-tmp/RoG/qwen/multimodels/google/gemma")
    elif args.model == "llama":
        model = llamamod(modelpath="/root/autodl-tmp/RoG/qwen/multimodels/meta-llama/llama")
    res=generate(model=model, dataset=ds)
    result_entry = {
        "model": args.model,
        "dataset": args.dataset,
        "timelist": res  #
    }
    with open("./timelist.jsonl", "a") as f:
        f.write(json.dumps(result_entry) + "\n")
