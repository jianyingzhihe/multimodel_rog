import json
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import argparse
import utils
from datasets import load_dataset
import datasets

datasets.disable_progress_bar()
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import torch
import re
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel
from loader import *

N_CPUS = (
    int(os.environ["SLURM_CPUS_PER_TASK"]) if "SLURM_CPUS_PER_TASK" in os.environ else 1
)
PATH_RE = r"<PATH>(.*)<\/PATH>"
INSTRUCTION = """Given the image and question below, please generate a valid relation path (in <PATH>...</PATH> format) that can help answer the question using a knowledge graph.

Example:
Question: What sport can you use this for?
Answer: <PATH> vehicle.brand <SEP> vehicle.type <SEP> sports.use </PATH>

Now answer:
"""
REWARD="now you know the information about the image:"
FOLLOWS=",please answer the questions:"
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


def parse_prediction(prediction):
    """
    Parse a list of predictions to a list of rules

    Args:
        prediction (_type_): _description_

    Returns:
        _type_: _description_
    """
    results = []
    for p in prediction:
        path = re.search(PATH_RE, p)
        if path is None:
            continue
        path = path.group(1)
        path = path.split("<SEP>")
        if len(path) == 0:
            continue
        rules = []
        for rel in path:
            rel = rel.strip()
            if rel == "":
                continue
            rules.append(rel)
        results.append(rules)
    return results


def generate_seq(
    model, question, pictpath,num_beam=3, do_sample=False, max_new_tokens=100
):
    scores=[]
    prediction = []
    norm_scores = []
    # tokenize the question
    question=INSTRUCTION+question
    output = model.infwithscore(question,pictpath)


    if num_beam > 1:
        for each in output:
            prediction.append(each["answer"])
            scores.append(each["score"])
            norm_scores.append(each["normalized_score"])
    else:
        scores = [1]
        norm_scores = [1]

    return {"paths": prediction, "scores": scores, "norm_scores": norm_scores}


def generatembody(ans):
    results=[]
    for s in ans:
        s=s.replace("Answer:", "").strip()
        match = re.search(r'<PATH>(.*?)</PATH>', s)
        if match:
            parts = [item.strip() for item in match.group(1).split('<SEP>')]
            results.append(parts)
    return results


def gen_prediction(args):
    print(args)
    model = qwenmod()
    okvqadata = datas(args.data_path)

    output_dir = os.path.join(args.output_path, args.d, args.model_name, args.split)
    print("Save results to: ", output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prediction_file = os.path.join(output_dir, f"predictions_{args.n_beam}_{args.do_sample}.jsonl")

    with open(prediction_file, "w") as f:
        for data in tqdm(okvqadata.combined):
            question = INSTRUCTION + data["question"]
            qid = data["id"]
            pictpath = data["image_path"]

            raw_output = generate_seq(
                model,
                question,
                pictpath,
                max_new_tokens=args.max_new_tokens,
                num_beam=args.n_beam,
                do_sample=args.do_sample,
            )
            raw_output["paths"] = generatembody(raw_output["paths"])
            tempscr=0
            for path,scr in zip(raw_output["paths"], raw_output["norm_scores"]):
                if scr > tempscr:
                    tempscr=scr
                    temppath=path
            raw_output["ground_paths"]=temppath
            res = {
                "prediction": raw_output['paths'],
                "id": qid,
                "input": question,
                'scores': raw_output['scores'],
                "norm_scores": raw_output['norm_scores'],
                "ground_paths": raw_output['ground_paths'],
            }

            # 使用 json.dumps 将字典转换为 JSON 字符串并写入文件
            f.write(json.dumps(res) + "\n")
            f.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="data/OKVQA"
    )
    parser.add_argument("--d", "-d", type=str, default="RoG-webqsp")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
    )
    parser.add_argument("--output_path", type=str, default="results/gen_rule_path")
    parser.add_argument(
        "--model_name",
        type=str,
        help="model_name for save results",
        default="Llama-2-7b-hf",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="model_name for save results",
        default="meta-llama/Llama-2-7b-chat-hf",
    )
    parser.add_argument(
        "--prompt_path", type=str, help="prompt_path", default="prompts/llama2.txt"
    )
    parser.add_argument(
        "--rel_dict",
        nargs="+",
        default=["datasets/KG/fbnet/relations.dict"],
        help="relation dictionary",
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="force to overwrite the results"
    )
    parser.add_argument("--debug", action="store_true", help="Debug")
    parser.add_argument("--lora", action="store_true", help="load lora weights")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--n_beam", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true", help="do sampling")

    args = parser.parse_args()

    gen_path = gen_prediction(args)
