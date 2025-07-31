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


def generate(model, dataset, outputdir, use_system=True, dataset_type=None):
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
        # 根据数据集类型选择正确的数据属性
        if dataset_type == "fvqa":
            data_source = dataset.val
        else:
            data_source = dataset.combined
            
        for each in tqdm.tqdm(data_source, desc="Processing images"):
            id = each.id
            if id in td:
                continue
            try:
                image = each.image
                question = each.question
                
                # 为 VLLM 构造 URL 格式的图像路径
                if model.type == "vllm":
                    # 构造完整的图像路径
                    if not os.path.isabs(image):
                        # 如果是相对路径，使用 image_path 作为基础路径
                        full_image_path = os.path.join(image_path, image)
                    else:
                        full_image_path = image
                    
                    # 确保路径存在
                    if not os.path.exists(full_image_path):
                        print(f"Warning: Image file not found: {full_image_path}")
                    
                    image_file_url = "file:///" + full_image_path
                    print(f"Image URL: {image_file_url}")
                    
                    if use_system:
                        messages = [
                            {"role":"system",
                             "content": "You are a visual reasoning assistant. Please first identify the objects and their attributes in the image, then construct a reasonable relationship path based on the question, and finally provide the answer in English."},
                            {"role": "user",
                             "content": [{"type": "image_url", "image_url": {"url": image_file_url}}, {"type": "text", "text": question}]}
                        ]
                    else:
                        messages = [
                            {"role": "user",
                             "content": [{"type": "image_url", "image_url": {"url": image_file_url}}, {"type": "text", "text": question}]}
                        ]
                else:
                    # 对于 HF 模式，使用原来的格式
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
        print(f"\nProcessed {len(td)} items")




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
    parser.add_argument("--auth_token", type=str, default=None,
                       help="Hugging Face authentication token for gated models")
    
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
        model = qwenmod(modelpath=args.modelpath, type=args.infer_type, allowed_local_media_path=image_path, use_auth_token=args.auth_token)
    elif args.modeltype=="gemma" or args.modeltype=="google":
        model=googlemod(modelpath=args.modelpath, type=args.infer_type, allowed_local_media_path=image_path, use_auth_token=args.auth_token)
    elif args.modeltype=="llama":
        model=llamamod(modelpath=args.modelpath, type=args.infer_type, allowed_local_media_path=image_path, use_auth_token=args.auth_token)
    elif args.modeltype=="intern":
        model=internmod(modelpath=args.modelpath, type=args.infer_type, allowed_local_media_path=image_path, use_auth_token=args.auth_token)

    generate(model=model, dataset=ds, outputdir=args.outputdir, use_system=args.system, dataset_type=args.dataset_type)
    # ds.evaluate_jsonl(args.outputdir)
