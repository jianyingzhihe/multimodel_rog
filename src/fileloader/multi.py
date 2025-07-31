# multimodal/base.py
from abc import ABC, abstractmethod

class BaseMultiModalModel(ABC):
    def __init__(self, modelpath: str = "./",type="hf", **kwargs):
        self.modelpath = modelpath
        self.type = type
        self._load_model(type=self.type, **kwargs)

    @abstractmethod
    def _load_model(self,type="vllm", **kwargs):
        """加载模型和处理器"""
        pass

    @abstractmethod
    def inf_question_image(self, question: str, image: str):
        """基础推理接口：输入问题和图像路径，返回文本答案"""
        pass

    @abstractmethod
    def inf_with_messages(self, messages: list):
        """支持对话历史的消息格式推理接口"""
        pass

    def inf_with_score(self, question: str, image: str, max_new_tokens=128, num_beam=3):
        """
        可选功能：支持束搜索生成多个答案及得分。
        默认抛出 NotImplementedError，子类可选择性实现。
        """
        raise NotImplementedError("infwithscore is not implemented for this model.")
