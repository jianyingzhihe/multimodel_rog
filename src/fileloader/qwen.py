# multimodal/qwenmod.py
import os
import torch
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import Qwen2_5_VLForConditionalGeneration
from .dataloader import *
from .multi import BaseMultiModalModel


class qwenmod(BaseMultiModalModel):
    def _load_model(self,type="hf"):
        self.modeltype="qwen"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.modelpath,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.modelpath)

    def inf_question_image(self, question: str, image: Image.Image):
        # 确保 image 是 PIL.Image 对象
        if not isinstance(image, Image.Image):
            raise ValueError("Expected a PIL.Image object")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]
        return self.inf_with_messages(messages)

    def inf_with_messages(self, messages: list):
        # 处理文本部分
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 提取并处理图像信息
        image_inputs, video_inputs = [], []
        for message in messages:
            for content in message.get('content', []):
                if content['type'] == 'image':
                    if isinstance(content['image'], str):  # 如果是字符串，则认为是文件路径
                        image = Image.open(content['image']).convert('RGB')
                    elif isinstance(content['image'], Image.Image):  # 如果是 PIL.Image 对象
                        image = content['image']
                    else:
                        raise ValueError("Unsupported image type")
                    image_inputs.append(image)
        print(text)
        print(image_inputs)

        # 使用 processor 将文本和图像转换为模型输入
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda").to(torch.bfloat16)  # 使用 float16

        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        return output_text

    def inf_with_score(self, question: str, pictpath: str, max_new_tokens=128, num_beams=3):
        if not os.path.exists(pictpath):
            raise FileNotFoundError(f"Image file not found: {pictpath}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pictpath},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        image = Image.open(pictpath).convert("RGB")
        image.thumbnail((512, 512))  # 缩放以减少显存占用

        inputs = self.processor(
            text=[text],
            images=image,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda").to(torch.bfloat16)  # 使用 float16

        print("Generating with beam search...")

        output = self.model.generate(
            **inputs,
            do_sample=False,  # 不采样，使用束搜索
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_beam_groups=num_beams,  # 启用分组束搜索
            diversity_penalty=1.0,      # 多样性惩罚
            num_return_sequences=num_beams,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True
        )

        print("Generate finished.")

        generated_ids_trimmed = [out_ids[len(inputs.input_ids[0]):] for out_ids in output.sequences]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

        scores = output.sequences_scores.tolist()
        norm_scores = torch.softmax(output.sequences_scores, dim=0).tolist()

        results = [
            {
                "answer": output_text[i].strip(),
                "score": scores[i],
                "normalized_score": norm_scores[i]
            } for i in range(len(output_text))
        ]

        return results