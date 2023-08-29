#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   fastllm_predictor.py
@Time    :   2023/08/23 12:55:35
@Author  :   nicholas wu 
@Version :   1.0
@Contact :   nicholas_wu@aliyun.com
@License :    
@Desc    :   None
'''
import os
import re
from typing import List, Dict
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from peft import PeftModel
from fastllm_pytools import llm
from .utils import split_paragraph_into_many, build_doccano_label


class Predictor(object):
    def __init__(self, model_name_or_path: str, extra_model_name_or_path: str = None, *args, **kwargs) -> None:
        self._load_model(model_name_or_path, extra_model_name_or_path, *args, **kwargs)

    def _load_model(self, model_name_or_path, extra_model_name_or_path, *args, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, extra_model_name_or_path)
        model = model.eval()
        model = llm.from_hf(model, self.tokenizer, dtype="int4")
        self.model = model

    def get_prompt(self, text, question=None, task=None, history=[]):
        if task == "ee":
            text = f'''基于以下已知信息，完成事件抽取任务，如果无法从中得到答案，只能回复\"根据已知信息无法回答该问题\"。\n已知信息：{text}\n问题：请从已知信息中抽取属于\"风险事件类型集合S\"的风险事件类型s，并抽取每个风险事件类型s对应的机构名称。\n输出要求：每行以\"事件类型-机构名称1;机构名称2...\"的格式呈现，不同风险事件类型用\"\n\"分割。'''
        elif task == "dqa":
            text = f'''基于以下已知信息，简洁和专业的来回答用户的问题，如果无法从中得到答案，请说 "根据已知信息无法回答该问题"。\n已知信息：{text}\n问题：{question}'''
        elif task == "summary":
            text = f'''基于以下已知信息，简洁和专业的来回答用户的问题。\n已知信息：{text}\n问题：请生成关于已知信息的摘要。'''
        elif task == "sim_gen":
            text = f"""帮我生成`{text}`的5个表述各异但含义相同的相似问题"""
        return text

    def batch_predict(self, batch, question=None, task=None, history=[], **kwargs):
        batch = [self.get_prompt(text, question, task) for text in batch]
        response = self.model.batch_chat(self.tokenizer, batch, **kwargs)
        return response

    def predict(self, text, question=None, task=None, history=[], **kwargs):
        text = self.get_prompt(text, question, task, history)
        response, history = self.model.chat(self.tokenizer, text, history=history, **kwargs)
        return response, history
        

class PoPredictor(Predictor):
    def __init__(self, model_name_or_path: str, ptuning_checkpoint: str = None, pre_seq_len: int = 128, quantize: bool = False) -> None:
        super().__init__(model_name_or_path, ptuning_checkpoint, pre_seq_len, quantize)
        self.regex = re.compile(".*?-")

    def _pre_process(self, text: str, paragraphing: bool=False) -> List[str]:
        if paragraphing:
            texts = split_paragraph_into_many(text, max_length=256)
        else:
            texts = [text]
        return texts

    def _predict(self, texts, question=None, task=None, history=[], **kwargs) -> List[str]:
        responses = []
        for text in texts:
            text = self.get_prompt(text, question, task, history)
            label = self.model.response(text, history=history, **kwargs)
            responses.append(label)
        return responses

    def _get_label_pairs(self, text):
        if not text or "-" not in text:
            return []
        labels=[]
        for i in text.split("\n"):
            label = self.regex.search(i).group()[:-1]
            words = i[len(label)+1:]
            # 去重
            words = list(set(words.split(";")))
            for word in words:
                labels.append(label + "-" + word)
        return labels

    def _post_process(self, text: str):
        text = text.strip()
        text = text.replace("nan", "").replace("\n\n", "\n")
        text = text.replace("；", ";").replace("_x000D_", "")
        text = text.replace("（", "(").replace("）", ")")
        text = self._get_label_pairs(text)
        return text

    def response_output_wrapper(self, text, responses: List[Dict[str, str]], require_doccano_label) -> Dict:
        labels = []
        paragraphs = []
        for item in responses:
            labels.extend(item["labels"])
            if require_doccano_label:
                item["doccano_label"] = build_doccano_label(item["text"], item["labels"])
            paragraphs.append(item)
                
        labels = list(set(labels))
        
        response = {
                "text": text,
                "labels": labels,
                "paragraphs": paragraphs
            }
        return response

    def predict(self, text, task="ee", **kwargs) -> Dict:
        paragraphing = kwargs["paragraphing"]
        require_doccano_label = kwargs["require_doccano_label"]
        texts = self._pre_process(text, paragraphing)
        responses = self._predict(texts, task=task)
        responses = [{"text": text, "labels": self._post_process(response)} for text, response in zip(texts, responses)]
        response = self.response_output_wrapper(text, responses, require_doccano_label)
        return response