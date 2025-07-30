import json
import argparse
import tqdm
import os
from distutils.util import strtobool
os.environ["TORCHDYNAMO_DISABLE"] = "1"
from src.fileloader import dataf, qwenmod,datap,datas,googlemod,llamamode,internmod


def generate(model, dataset, outputdir, use_system=True):
    """
    使用给定的模型对数据集中的每个条目进行推理，并将结果保存为jsonl格式的文件。
    :param model: 已加载的模型实例。
    :param dataset: 数据集实例。
    :param outputdir: 输出目录路径。
    :param use_system: 是否使用system prompt。
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
                
                if use_system:
                    messages = [
                        {"role":"system",
                         "content":[
                             {"type":"text","text":"You are a visual reasoning assistant. Please first identify the objects and their attributes in the image, then construct a reasonable relationship path based on the question, and finally provide the answer in English."},
                         ]},
                        {"role": "user",
                         "content": [{"type": "image","image":image}, {"type": "text", "text": question}]}
                    ]
                else:
                    messages = [
                        {"role": "user",
                         "content": [{"type": "image","image":image}, {"type": "text", "text": question}]}
                    ]

                if model.modeltype == "intern":
                    if use_system:
                        messages="[{\"role\":\"system\",\"content\":[{\"type\":\"text\",\"text\":\"You are a visual reasoning assistant. Please first identify the objects and their attributes in the image, then construct a reasonable relationship path based on the question, and finally provide the answer in English.\"}]},{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\""+question+"\"}]}]"
                    else:
                        messages="[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\""+question+"\"}]}]"
                    result=model.infer(messages,image)
                else :
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




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal model validation script")
    parser.add_argument("--dataset_type", type=str, default="fvqa", 
                       choices=["fvqa", "aokvqa", "okvqa"],
                       help="Type of dataset to use")
    parser.add_argument("--modeltype", type=str, default="intern",
                       choices=["qwen", "gemma", "google", "llama", "intern"],
                       help="Type of model to use")
    parser.add_argument("--modelpath", type=str, default="./multimodels/Qwen/qwenvl",
                       help="Path to the model")
    parser.add_argument("--outputdir", type=str, default="./output_with_system_token_qwen_AOKVQA.jsonl",
                       help="Output directory for results")
    parser.add_argument("--system", type=lambda x: bool(strtobool(x)), default=True,
                       help="Whether to use system prompt (true/false)")
    parser.add_argument("--infer_type", type=str, default="hf",
                       choices=["hf", "vllm"],
                       help="Inference type to use")
    
    args = parser.parse_args()
    
    #加载不同数据集
    if args.dataset_type=="fvqa":
        qapath="/home/z_wen/Kbvqa/data/FVQA/new_dataset_release/all_qs_dict_release.json"
        image="/home/z_wen/Kbvqa/data/FVQA/new_dataset_release/images"
        ds=dataf(qapath,image)
    if args.dataset_type=="aokvqa":
        ds=datap("/root/autodl-tmp/RoG/qwen/data/AOKVQA/data/test-00000-of-00001-d306bf3ad53b6618.parquet")
    if args.dataset_type=="okvqa":
        ds=datas("/home/z_wen/Kbvqa/data/okvqa_data/val/")

    #加载不同模型
    if args.modeltype=="qwen":
        model = qwenmod(args.modelpath, type=args.infer_type)
    elif args.modeltype=="gemma" or args.modeltype=="google":
        model=googlemod(args.modelpath, type=args.infer_type)
    elif args.modeltype=="llama":
        model=llamamode(args.modelpath, type=args.infer_type)
    elif args.modeltype=="intern":
        model=internmod(args.modelpath)

    generate(model=model, dataset=ds, outputdir=args.outputdir, use_system=args.system)
    # ds.evaluate_jsonl(args.outputdir)
