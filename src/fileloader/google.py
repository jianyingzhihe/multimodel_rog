# multimodal/qwenmod.py
import os
import torch
from PIL import Image
from PIL.JpegImagePlugin import samplings
from transformers import AutoProcessor
# from qwen_vl_utils import process_vision_info
from modelscope import Gemma3ForConditionalGeneration
from .dataloader import *
from .multi import BaseMultiModalModel
from vllm import LLM,SamplingParams
from vllm.sampling_params import BeamSearchParams

class googlemod(BaseMultiModalModel):
    def _load_model(self,type="hf",max_tokens=512,allowed_local_media_path=None, use_auth_token=None, **kwargs):
        self.type=type
        self.modeltype="gemma"
        if type=="hf":
            print(os.getcwd())
            print(self.modelpath)
            
            # Try to load with flash attention 2 first, fallback to eager attention if it fails
            try:
                print("Attempting to load with flash_attention_2...")
                self.model = Gemma3ForConditionalGeneration.from_pretrained(
                    pretrained_model_name_or_path=self.modelpath,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                    token=use_auth_token if use_auth_token else None
                )
                print("Successfully loaded with flash_attention_2")
            except Exception as e:
                print(f"Flash attention 2 failed: {e}")
                print("Falling back to eager attention...")
                try:
                    self.model = Gemma3ForConditionalGeneration.from_pretrained(
                        pretrained_model_name_or_path=self.modelpath,
                        torch_dtype=torch.bfloat16,
                        attn_implementation="eager",
                        device_map="auto",
                        token=use_auth_token if use_auth_token else None
                    )
                    print("Successfully loaded with eager attention")
                except Exception as e2:
                    print(f"Eager attention also failed: {e2}")
                    print("Trying default attention implementation...")
                    self.model = Gemma3ForConditionalGeneration.from_pretrained(
                        pretrained_model_name_or_path=self.modelpath,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        token=use_auth_token if use_auth_token else None
                    )
                    print("Successfully loaded with default attention")
            
            self.processor = AutoProcessor.from_pretrained(self.modelpath, token=use_auth_token if use_auth_token else None)
        if type=="vllm":
            num_gpus = torch.cuda.device_count()
            self.sampling_params = SamplingParams(
                max_tokens=max_tokens,  # 修改这里：增加最大输出 token 数量

            )
            vllm_kwargs = {
                "model": self.modelpath,
                "tensor_parallel_size": num_gpus,
                "trust_remote_code": True,
                "tokenizer_mode": "auto",
                "dtype": torch.bfloat16,
                "enable_prefix_caching": True,
                "gpu_memory_utilization": 0.9,
                "allowed_local_media_path": allowed_local_media_path or "/root/autodl-tmp/RoG/qwen/data/OKVQA/val2014",
                "limit_mm_per_prompt": {"image": 1,"video": 0},
                "max_model_len": 4096,
                "max_num_seqs": 2
            }
            if use_auth_token:
                vllm_kwargs["hf_token"] = use_auth_token
            
            self.model = LLM(**vllm_kwargs)

    def inf_question_image(self, question: str, image: str):
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
        if self.type=="hf":
            try:
                inputs = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt"
                ).to(self.model.device, dtype=torch.bfloat16)
                input_len = inputs["input_ids"].shape[-1]
                
                with torch.inference_mode():
                    # Use more conservative generation parameters to avoid errors
                    generation = self.model.generate(
                        **inputs, 
                        max_new_tokens=512, 
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True
                    )
                    generation = generation[0][input_len:]
                decoded = self.processor.decode(generation, skip_special_tokens=True)
                return decoded
            except Exception as e:
                print(f"Generation failed with error: {e}")
                return f"Error during generation: {str(e)}"
        if self.type=="vllm":
            outputs = self.model.chat(messages, sampling_params=self.sampling_params)
            return outputs[0].outputs[0].text


    def inf_with_score(self, question: str, pictpath: str, max_new_tokens=512, num_beams=3):
        """
        通过给定的问题和图片路径，使用模型生成答案，并返回生成的答案及其置信度得分。
        :param question: 用户提出的问题，字符串类型。
        :param pictpath: 图片文件的路径，字符串类型。
        :param max_new_tokens: 模型生成文本的最大长度，默认为128。
        :param num_beams: 束搜索的数量，默认为3。
        :return: 包含生成答案及其对应分数的列表。
        """

        if not os.path.exists(pictpath):
            raise FileNotFoundError(f"Image file not found: {pictpath}")

        # 创建消息内容
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pictpath},
                    {"type": "text", "text": question}
                ]
            }
        ]

        if self.type == "hf":
            inputs = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image = Image.open(pictpath).convert("RGB")
            image.thumbnail((512, 512))  # 缩放以减少显存占用

            # 准备模型输入
            inputs = self.processor(
                text=[inputs],
                images=image,
                return_tensors="pt"
            ).to(self.model.device).to(torch.bfloat16)  # 确保与模型设备一致

            print("Generating with beam search...")

            # 使用束搜索生成答案
            output = self.model.generate(
                **inputs,
                do_sample=False,  # 不采样，使用束搜索
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True
            )

            print("Generate finished.")

            # 解码生成的token
            generated_ids_trimmed = [out_ids[len(inputs.input_ids[0]):] for out_ids in output.sequences]
            output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

            # 计算每个生成序列的得分
            scores = output.sequences_scores.tolist()
            norm_scores = torch.softmax(output.sequences_scores, dim=0).tolist()

            # 构建结果列表
            results = [
                {
                    "answer": output_text[i].strip(),
                    "score": scores[i],
                    "normalized_score": norm_scores[i]
                } for i in range(len(output_text))
            ]

            return results

        elif self.type == "vllm":
            pictpath="file://"+os.path.join("/root/autodl-tmp/RoG/qwen",pictpath)
            params = BeamSearchParams(beam_width=num_beams, max_tokens=max_new_tokens)
            outputs = self.model.beam_search(
                beam_width=num_beams,
                max_tokens=max_new_tokens,
                prompts=question,
                image= pictpath,
            )
            res=[]
            for out in outputs:
                res.append(out.sequence[0].text)
            print(res)
            return res
