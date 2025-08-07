import math
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams, EngineArgs

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


from .multi import BaseMultiModalModel

class internmod(BaseMultiModalModel):
    def _load_model(self, type="hf", max_tokens=512, allowed_local_media_path=None, use_auth_token=None, **kwargs):
        self.modeltype = "intern"
        self.type = type
        
        if type == "hf":
            self.device_map = split_model(self.modelpath)
            self.model = AutoModel.from_pretrained(
                self.modelpath,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                load_in_8bit=False,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map=self.device_map).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.modelpath, trust_remote_code=True, use_fast=False)
        
        elif type == "vllm":
            # 如果提供了 auth token，设置环境变量
            if use_auth_token:
                import os
                os.environ["HF_TOKEN"] = use_auth_token
            
            # VLLM 配置
            vllm_kwargs = {
                "model": self.modelpath,
                "trust_remote_code": True,
                "max_model_len": 16384,
                "limit_mm_per_prompt": {"image": 1, "video": 0},
                "tensor_parallel_size": torch.cuda.device_count(),
                "gpu_memory_utilization": 0.9,
            }
            
            if allowed_local_media_path:
                vllm_kwargs["allowed_local_media_path"] = allowed_local_media_path
            if use_auth_token:
                vllm_kwargs["hf_token"] = use_auth_token
                
            self.model = LLM(**vllm_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(self.modelpath, trust_remote_code=True)
            
            # 设置采样参数
            self.sampling_params = SamplingParams(
                max_tokens=max_tokens
            )
            
            # 设置停止词
            stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
            stop_token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in stop_tokens]
            stop_token_ids = [token_id for token_id in stop_token_ids if token_id is not None]
            self.sampling_params.stop_token_ids = stop_token_ids

    def infer(self, image, question):
        if self.type == "hf":
            pixel_values = load_image(image, max_num=12).to(torch.bfloat16).cuda()
            generation_config = dict(max_new_tokens=1024, do_sample=True)
            # 为图像问题添加 <image> 标记
            question_with_image = f'<image>\n{question}'
            result = self.model.chat(self.tokenizer, pixel_values, question_with_image, generation_config)
            return result
        
        elif self.type == "vllm":
            # 为图像问题添加 <image> 标记
            question_with_image = f'<image>\n{question}'
            
            # 构造消息格式，包含图像信息
            messages = [{
                "role": "user", 
                "content": [
                    {"type": "image_url", "image_url": {"url": f"file:///{image}"}},
                    {"type": "text", "text": question}
                ]
            }]
            
            # 使用 tokenizer 的 chat template
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # VLLM 推理 - 使用 chat 方法而不是 generate
            outputs = self.model.chat(messages, self.sampling_params)
            result = outputs[0].outputs[0].text
            return result
    
    def inf_question_image(self, question: str, image: str):
        """基础推理接口：输入问题和图像路径，返回文本答案"""
        return self.infer(image, question)
    
    def inf_with_messages(self, messages: list):
        """支持对话历史的消息格式推理接口"""
        if self.type == "vllm":
            # 对于 VLLM，直接使用原始消息格式，VLLM 会处理图像
            # 但需要确保图像 URL 格式正确
            processed_messages = []
            
            for message in messages:
                if message.get("role") == "system":
                    # 保持 system message 不变
                    processed_messages.append(message)
                elif message.get("role") == "user":
                    content = message.get("content", [])
                    if isinstance(content, list):
                        processed_content = []
                        
                        for item in content:
                            if item.get("type") == "text":
                                processed_content.append(item)
                            elif item.get("type") == "image":
                                # 转换为 image_url 格式
                                image_path = item.get("image")
                                processed_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"file:///{image_path}"}
                                })
                            elif item.get("type") == "image_url":
                                # 已经是正确格式，直接使用
                                processed_content.append(item)
                        
                        processed_messages.append({
                            "role": "user",
                            "content": processed_content
                        })
            
            # VLLM 推理 - 使用 chat 方法
            outputs = self.model.chat(processed_messages, self.sampling_params)
            return outputs[0].outputs[0].text
        
        else:
            # 对于 HF 模式，使用原来的逻辑
            question = ""
            image = None
            system_prompt = ""
            
            for message in messages:
                if message.get("role") == "system":
                    # 处理 system prompt
                    content = message.get("content", "")
                    if isinstance(content, str):
                        system_prompt = content
                    elif isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                system_prompt = item.get("text", "")
                elif message.get("role") == "user":
                    content = message.get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                question = item.get("text", "")
                            elif item.get("type") == "image":
                                image = item.get("image")
                            elif item.get("type") == "image_url":
                                image_url = item.get("image_url", {})
                                url = image_url.get("url", "")
                                if url.startswith("file:///"):
                                    image = url[8:]  # 移除 "file:///" 前缀
                                else:
                                    image = url
            
            # 如果有 system prompt，将其添加到问题前面
            if system_prompt:
                question = f"{system_prompt}\n\n{question}"
            
            if image and question:
                return self.infer(image, question)
            else:
                raise ValueError(f"Could not extract image and question from messages. Image: {image}, Question: {question}")

