import json

import tqdm
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
from src.fileloader import dataf, qwenmod,datap,datas


def generate(model, dataset, outputdir):
    """
    使用给定的模型对数据集中的每个条目进行推理，并将结果保存为jsonl格式的文件。
    :param model: 已加载的模型实例。
    :param dataset: 数据集实例。
    :param outputdir: 输出目录路径。
    """
    output_path = outputdir
    td=[]
    with open(outputdir) as f:
        for line in f:
            temp=json.loads(line)
            id=temp["id"]
            td.append(id)
    print(td)
    with open(output_path, 'a', encoding='utf-8') as f:
        batch=[]
        for each in tqdm.tqdm(dataset.combined, desc="Processing images"):
            id = each.id
            if id in td:
                continue
            try:
                image = each.image
                question = each.question
                messages = [
                    {"role":"system",
                     "content":[
                         {"type":"text","text":"You are a visual reasoning assistant. Please first identify the objects and their attributes in the image, then construct a reasonable relationship path based on the question, and finally provide the answer in English."},
                     ]},
                        {"role": "user",
                         "content": [{"type": "image","image":image}, {"type": "text", "text": {question}}]}
                    ]
                result = model.inf_with_messages(messages)
                output_dict = {
                            "id": id,
                            "question": question,
                            "answer": result
                        }
                f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:
                print(id)
                batch.append(id)
                print(e)
        print(batch)



# 示例调用
if __name__ == "__main__":
    dataset_type="fvqa"
    if dataset_type=="fvqa":
        qapath="/root/autodl-tmp/RoG/qwen/data/FVQA/new_dataset_release/all_qs_dict_release.json"
        image="/root/autodl-tmp/RoG/qwen/data/FVQA/new_dataset_release/images"
        ds=dataf(qapath,image)
    if dataset_type=="aokvqa":
        ds=datap("/root/autodl-tmp/RoG/qwen/data/AOKVQA/data/test-00000-of-00001-d306bf3ad53b6618.parquet")
    if dataset_type=="okvqa":
        ds=datas("/root/autodl-tmp/RoG/qwen/data/OKVQA/")
    modelpath = "./multimodels/Qwen/qwenvl"
    outputdir = "./output_with_system_token_qwen_AOKVQA.jsonl"  # 指定输出目录
    model = qwenmod(modelpath,type="hf")
    generate(model=model, dataset=ds, outputdir=outputdir)
    # ds.evaluate_jsonl(outputdir)
