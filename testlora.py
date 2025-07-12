import json
import tqdm
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
from src.fileloader.google import datas, googlemod

def generate(model, dataset, outputdir):
    """
    使用给定的模型对数据集中的每个条目进行推理，并将结果保存为jsonl格式的文件。
    :param model: 已加载的模型实例。
    :param dataset: 数据集实例。
    :param outputdir: 输出目录路径。
    """
    output_path = outputdir
    with open(output_path, 'w+', encoding='utf-8') as f:
        for each in tqdm.tqdm(dataset.combined, desc="Processing images"):
            id = each["id"]
            image_path = each["image_path"]
            question = each["question"]
            messages = [
                {"role": "user",
                 "content": [{"type": "image", "image": image_path}, {"type": "text", "text": question}]}
            ]
            result = model.inf_with_messages(messages=messages)
            output_dict = {
                "id": id,
                "question": question,
                "answer": result
            }

            f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")
            f.flush()



modelpath = "/root/autodl-tmp/RoG/qwen/output/v11-20250710-145738/checkpoint-201"
outputdir = "./gemma_basic.jsonl"  # 指定输出目录
dataset = datas(datapath)
model = googlemod(modelpath,type="hf")
generate(model=model, dataset=dataset, outputdir=outputdir)
dataset.evaluate_jsonl(outputdir)