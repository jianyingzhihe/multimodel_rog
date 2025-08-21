import traceback
import os
import argparse
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from swift.llm import (
    PtEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template, InferRequest
)
from swift.tuners import Swift
from src.fileloader.dataloader import *

def solve(daset, engine):
    cnt=0
    timelist=[]
    t0=time.time()
    if daset.datatype=="okvqa":
        for each in tqdm.tqdm(daset.combined):
            cnt+=1
            infer_request = InferRequest(
                    messages=[
                        {"role": "system",
                         "content": "You are a visual reasoning assistant. Please first identify the objects and their attributes in the image, then construct a reasonable relational path based on the question, and finally provide the answer."},
                        {"role": "user", "content": f"<image>{each.question}"}  # 使用实际问题
                    ],
                    images=[each.image]
                )
            try:
                resp_list = engine.infer([infer_request], request_config)
                response_text = resp_list[0].choices[0].message.content
                result = {
                        "id": each.id,
                        "question": each.question,
                        "image_path": each.image,
                        "predicted_answer": response_text,
                    }
            except:
                traceback.print_exc()
            total_time = time.time() - t0
            average_time = total_time / cnt
            timelist.append({"iteration": cnt, "total_time": total_time, "average_time": average_time})
            if cnt==500:
                return timelist
    elif daset.datatype=="fvqa":
        for each in tqdm.tqdm(daset.val):
            cnt+=1
            infer_request = InferRequest(
                    messages=[
                        {"role": "system",
                         "content": "You are a visual reasoning assistant. Please first identify the objects and their attributes in the image, then construct a reasonable relational path based on the question, and finally provide the answer."},
                        {"role": "user", "content": f"<image>{each.question}"}  # 使用实际问题
                    ],
                    images=[each.image]
                )
            try:
                resp_list = engine.infer([infer_request], request_config)
                response_text = resp_list[0].choices[0].message.content
                result = {
                        "id": each.id,
                        "question": each.question,
                        "image_path": each.image,
                        "predicted_answer": response_text,
                    }
            except:
                traceback.print_exc()
            total_time = time.time() - t0
            average_time = total_time / cnt
            timelist.append({"iteration": cnt, "total_time": total_time, "average_time": average_time})
            if cnt==500:
                return timelist
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--dataset")

    args = parser.parse_args()
    if args.dataset=="okvqa":
        ds=datas("/root/autodl-tmp/RoG/qwen/data/OKVQA",split="val")
    elif args.dataset=="fvqa":
        qapath = "/root/autodl-tmp/RoG/qwen/data/FVQA/new_dataset_release/all_qs_dict_release.json"
        image = "/root/autodl-tmp/RoG/qwen/data/FVQA/new_dataset_release/images"
        ds = dataf(qapath, image)
    if args.model=="llama":
        model = '/root/autodl-tmp/RoG/qwen/multimodels/meta-llama/llama'
        if args.dataset=="fvqa":
            lora_checkpoint = safe_snapshot_download('/root/autodl-tmp/RoG/qwen/output-lora/fvqa/llama/roglora_fvqa_llama/checkpoint-867')
        elif args.dataset=="okvqa":
            lora_checkpoint = safe_snapshot_download('/root/autodl-tmp/RoG/qwen/output-lora/okvqa/llama/rog_okvqa_en_llama/checkpoint-4971')
    elif args.model=="gemma":
        model="/root/autodl-tmp/RoG/qwen/multimodels/google/gemma"
        if args.dataset=="fvqa":
            lora_checkpoint = safe_snapshot_download('/root/autodl-tmp/RoG/qwen/output-lora/fvqa/gemma/roglora_fvqa_gemma/checkpoint-882')
        elif args.dataset=="okvqa":
            lora_checkpoint = safe_snapshot_download('/root/autodl-tmp/RoG/qwen/output-lora/okvqa/gemma/rogokvqaengemma/checkpoint-4971')
    elif args.model=="qwen":
        model="/root/autodl-tmp/RoG/qwen/multimodels/Qwen/qwenvl"
        if args.dataset == "fvqa":
            lora_checkpoint = safe_snapshot_download('/root/autodl-tmp/RoG/qwen/output-lora/fvqa/qwen/rog_qwen_fvqa/checkpoint-867')
        elif args.dataset == "okvqa":
            lora_checkpoint = safe_snapshot_download('/root/autodl-tmp/RoG/qwen/output-lora/okvqa/qwen/rogokvqaqwen/checkpoint-2229')

    template_type = None
    default_system = None
    model, tokenizer = get_model_tokenizer(model)
    model = Swift.from_pretrained(model, lora_checkpoint)
    template_type = template_type or model.model_meta.template
    template = get_template(template_type, tokenizer, default_system=default_system)
    engine = PtEngine.from_model_template(model, template, max_batch_size=2)
    request_config = RequestConfig(max_tokens=2048, temperature=0)
    res=solve(ds, engine)
    result_entry = {
        "model": args.model,
        "dataset": args.dataset,
        "timelist": res  #
    }
    with open("./timelist.jsonl", "a") as f:
        f.write(json.dumps(result_entry) + "\n")