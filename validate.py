import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# os.environ["VLLM_LOGGING_LEVEL"]="DEBUG"
# os.environ["NCCL_DEBUG"]="TRACE"
# os.environ["VLLM_TRACE_FUNCTION"]="1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import json
import argparse
import tqdm

from distutils.util import strtobool

from src.fileloader import dataf, qwenmod, datap, datas, googlemod, llamamod, internmod


def generate(model, dataset, outputdir, use_system=True):
    """
    使用给定的模型对数据集中的每个条目进行推理，并将结果保存为jsonl格式的文件。
    :param model: 已加载的模型实例。
    :param dataset: 数据集实例。
    :param outputdir: 输出目录路径。
    :param use_system: 是否使用system prompt。
    """
    output_path = outputdir
    # 确保输出目录存在
    output_dir = os.path.dirname(outputdir)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    td=[]
    # 检查输出文件是否存在，如果存在则读取已处理的ID
    if os.path.exists(outputdir):
        with open(outputdir) as f:
            for line in f:
                temp=json.loads(line)
                id=temp["id"]
                td.append(id)
        print(f"Found {len(td)} already processed items")
    else:
        print("Output file does not exist, starting from scratch")
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
    parser.add_argument("--modeltype", type=str, default="qwen",
                       choices=["qwen", "gemma", "google", "llama", "intern"],
                       help="Type of model to use")
    parser.add_argument("--modelpath", type=str, default="Qwen/Qwen2.5-VL-72B-Instruct",
                       help="Path to the model")
    parser.add_argument("--outputdir", type=str, default="/home/z_wen/Kbvqa/multimodal_rog/res/output_no_system_qwen72b_fvqa.jsonl",
                       help="Output directory for results")
    parser.add_argument("--system", type=lambda x: bool(strtobool(x)), default=False,
                       help="Whether to use system prompt (true/false)")
    parser.add_argument("--infer_type", type=str, default="vllm",
                       choices=["hf", "vllm"],
                       help="Inference type to use")
    
    args = parser.parse_args()
    
    #加载不同数据集
    image_path = None
    if args.dataset_type=="fvqa":
        qapath="/home/z_wen/Kbvqa/data/FVQA/new_dataset_release/all_qs_dict_release.json"
        image_path="/home/z_wen/Kbvqa/data/FVQA/new_dataset_release/images"
        ds=dataf(qapath,image_path)
    if args.dataset_type=="aokvqa":
        image_path="/root/autodl-tmp/RoG/qwen/data/AOKVQA/val2014"  # AOKVQA 图像路径
        ds=datap("/root/autodl-tmp/RoG/qwen/data/AOKVQA/data/test-00000-of-00001-d306bf3ad53b6618.parquet")
    if args.dataset_type=="okvqa":
        image_path="/home/z_wen/Kbvqa/data/okvqa_data/val/val2014"  # OKVQA 图像路径
        ds=datas("/home/z_wen/Kbvqa/data/okvqa_data/val/")

    #加载不同模型
    if args.modeltype=="qwen":
        model = qwenmod(args.modelpath, type=args.infer_type, allowed_local_media_path=image_path)
    elif args.modeltype=="gemma" or args.modeltype=="google":
        model=googlemod(args.modelpath, type=args.infer_type, allowed_local_media_path=image_path)
    elif args.modeltype=="llama":
        model=llamamod(args.modelpath, type=args.infer_type, allowed_local_media_path=image_path)
    elif args.modeltype=="intern":
        model=internmod(args.modelpath)

    generate(model=model, dataset=ds, outputdir=args.outputdir, use_system=args.system)
    # ds.evaluate_jsonl(args.outputdir)
