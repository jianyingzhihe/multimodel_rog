import tqdm
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from src.fileloader.google import *
from swift.llm import (
    PtEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template, InferRequest
)
from swift.tuners import Swift
model = './multimodels/meta-llama/llama'
lora_checkpoint = safe_snapshot_download('/root/autodl-tmp/RoG/qwen/output-lora/v16-20250707-084334/checkpoint-2232')
template_type = None
default_system = None
model, tokenizer = get_model_tokenizer(model)
model = Swift.from_pretrained(model, lora_checkpoint)
template_type = template_type or model.model_meta.template
template = get_template(template_type, tokenizer, default_system=default_system)
engine = PtEngine.from_model_template(model, template, max_batch_size=2)
request_config = RequestConfig(max_tokens=1024, temperature=0)
daset = datas("./data/OKVQA", type="val")



def solve(daset, engine):
    output_file = "res/llama/output_roglora_results_ro2.jsonl"
    fw = open(output_file, 'w', encoding='utf-8')
    for each in tqdm.tqdm(daset.combined):
        # 构造请求
        infer_request = InferRequest(
            messages=[
                {"role": "system",
                 "content": "你是视觉推理助手。请先识别图像中的对象及其属性，然后根据问题构建合理的关系路径，最后给出答案。"},
                {"role": "user", "content": f"<image>{each['question']}"}  # 使用实际问题
            ],
            images=[each["image_path"]]
        )

        # 推理
        resp_list = engine.infer([infer_request], request_config)

        # 提取响应内容
        response_text = resp_list[0].choices[0].message.content

        # 构建输出字典
        result = {
            "id": each["id"],
            "question": each["question"],
            "image_path": each["image_path"],
            "predicted_answer": response_text,
            # 如果你有 ground truth 答案也可以加上
            # "ground_truth": each.get("answer", "")
        }

        # 写入 JSONL 文件
        fw.write(f"{json.dumps(result, ensure_ascii=False)}\n")

    fw.close()
    print(f"✅ 推理完成，结果已保存至 {output_file}")

def solve_vllm(daset, llm,lora_req):
    output_file = "res/gemma/google_roglora_results.jsonl"
    fw = open(output_file, 'w', encoding='utf-8')
    for each in tqdm.tqdm(daset.combined):
        # 构造请求
        image_path = each.image
        image_path = os.path.join("/root/autodl-tmp/RoG/qwen", image_path)
        image_file_url = "file:///" + image_path
        print(image_file_url)
        messages=[
                {"role": "system",
                 "content": "你是视觉推理助手。请先识别图像中的对象及其属性，然后根据问题构建合理的关系路径，最后给出答案。"},
                {"role": "user",
                 "content": [{"type": "image_url", "image_url": {"url":image_file_url}}, {"type": "text", "text": each.question}]}
            ],

        outputs = llm.chat(messages, lora_request=lora_req)
        res=outputs[0].outputs[0].text
        print(res)
        # 构建输出字典
        result = {
            "id": each.id,
            "question": each.question,
            "image_path": each.image,
            "predicted_answer": res,
            # 如果你有 ground truth 答案也可以加上
            # "ground_truth": each.get("answer", "")
        }

        # 写入 JSONL 文件
        fw.write(f"{json.dumps(result, ensure_ascii=False)}\n")

    fw.close()
    print(f"✅ 推理完成，结果已保存至 {output_file}")

solve(daset, engine)
