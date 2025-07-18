import time
import json
import os

import PIL
import tqdm
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
from src.fileloader.llama import datas, llamamod


def generate(model, dataset, outputdir):
    """
    使用给定的模型对数据集中的每个条目进行推理，并将结果保存为jsonl格式的文件。
    :param model: 已加载的模型实例。
    :param dataset: 数据集实例。
    :param outputdir: 输出目录路径。
    """
    output_path = outputdir
    with open(output_path, 'w', encoding='utf-8') as f:
        batch=[]
        for each in tqdm.tqdm(dataset.combined, desc="Processing images"):
            try:
                id = each["id"]
                image_path = each["image_path"]
                image_path=os.path.join("/root/autodl-tmp/RoG/qwen",image_path)
                print(image_path)

                question = each["question"]
                image_file_url="file:///"+image_path
                messages = [
                {"role": "system",
                 "content": "你是视觉推理助手。请先识别图像中的对象及其属性，然后根据问题构建合理的关系路径，最后给出答案。"},
                    {"role": "user",
                     "content": [{"type": "image_url", "image_url": {"url":image_file_url}}, {"type": "text", "text": question}]}
                ]


                result = model.inf_with_messages(messages=messages)

                    # 创建包含必要信息的字典
                output_dict = {
                        "id": id,
                        "question": question,
                        "answer": result
                    }

                # 将字典转换为JSON字符串并写入文件
                f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:
                print(e,id)


# 示例调用
datapath = "./data/OKVQA"
modelpath = "./multimodels/meta-llama/llama"
outputdir = "./output_with_system_token_llama.jsonl"  # 指定输出目录

dataset = datas(datapath)
model = llamamod(modelpath,type="vllm")

generate(model=model, dataset=dataset, outputdir=outputdir)
dataset.evaluate_jsonl(outputdir)