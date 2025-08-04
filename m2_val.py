import os
import torch

from modelscope import (
    AutoProcessor,
    AutoTokenizer,
)

import warnings
import argparse
from modeling_bailing_qwen2_5 import Bailing_qwen2_5NativeForConditionalGeneration
from processing_bailing_qwen2_5 import Bailing_qwen2_5Processor
import tqdm
import json


warnings.filterwarnings("ignore")
from .src.fileloader.dataloader import *

class BailingMMInfer:
    def __init__(self,
                 model_name_or_path,
                 device="cuda",
                 max_pixels=None,
                 min_pixels=None,
                 video_max_pixels=768 * 28 * 28,
                 video_min_pixels=128 * 28 * 28,
                 generation_config=None
                 ):
        super().__init__()
        self.model_name_or_path = model_name_or_path

        self.device = device

        self.device_map = device

        self.video_max_pixels = video_max_pixels if video_max_pixels is not None else 768 * 28 * 28
        self.video_min_pixels = video_min_pixels if video_min_pixels is not None else 128 * 28 * 28

        self.model, self.tokenizer, self.processor = self.load_model_processor()
        if max_pixels is not None:
            self.processor.max_pixels = max_pixels
        if min_pixels is not None:
            self.processor.min_pixels = min_pixels
        if generation_config is None:
            generation_config = {
                "num_beams": 1,
                "do_sample": True,
                "temperature": 0.9
            }

        self.generation_config = generation_config

    def load_model_processor(self):

        model = Bailing_qwen2_5NativeForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,

        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, add_bos_token=True, trust_remote_code=True)
        processor = Bailing_qwen2_5Processor.from_pretrained(self.model_name_or_path, trust_remote_code=True)

        return model, tokenizer, processor

    def generate(self, messages, max_new_tokens=8000):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, use_system=True
        )

        image_inputs, video_inputs = self.processor.process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )
        # print(inputs)
        print(self.tokenizer.decode(inputs['input_ids'][0]))

        inputs = inputs.to(self.device)

        for k in inputs.keys():
            if k == "pixel_values" or k == "pixel_values_videos":
                inputs[k] = inputs[k].to(dtype=torch.bfloat16)

        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                **self.generation_config,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        return output_text





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="/root/autodl-tmp/RoG/qwen/multimodels/m2")
    parser.add_argument('--max_pixels', type=int, default=401408)
    parser.add_argument('--min_pixels', type=int, default=401408)
    parser.add_argument('--max_new_tokens', type=int, default=8000)
    parser.add_argument('--data', type=str, default="/root/autodl-tmp/RoG/qwen/data/OKVQA")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_name_or_path = os.path.join(args.input_dir, args.model_name_or_path)

    model= BailingMMInfer(
        args.model_name_or_path,
        device=device,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels
    )
    dataset=datas(args.data)
    output_path="./output_with_system_token_m2.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for each in tqdm.tqdm(dataset.combined, desc="Processing images"):
            id = each["id"]
            image_path = each["image_path"]
            image_path = os.path.join("/root/autodl-tmp/RoG/qwen", image_path)
            print(image_path)
            question = each["question"]
            messages = [
                {"role": "system",
                 "content": [
                     {"type": "text",
                      "text":"你是视觉推理助手。请先识别图像中的对象及其属性，然后根据问题构建合理的关系路径，最后给出答案。"}]},
                {"role": "user",
                 "content": [
                     {"type": "image", "image": image_path},
                     {"type": "text",
                     "text": question}
                 ],
                 },
            ]
            print(messages)
            result = model.generate(messages=messages,max_new_tokens=args.max_new_tokens)

            # 创建包含必要信息的字典
            output_dict = {
                "id": id,
                "question": question,
                "answer": result
            }

            # 将字典转换为JSON字符串并写入文件
            f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")
            f.flush()