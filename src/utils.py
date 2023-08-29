#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2023/07/11 14:53:16
@Author  :   nicholas wu 
@Version :   1.0
@Contact :   nicholas_wu@aliyun.com
@License :    
@Desc    :   None
'''
import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import jsonlines
from typing import Dict


def read_jsonlines(path):
    res = []
    with open(path, 'r') as f:
        for item in jsonlines.Reader(f):
            res.append(item)
    return res


def write_jsonlines(data, path):
    with jsonlines.open(path, "w") as f:
        for i in tqdm(data):
            f.write(i)


def split_paragraph_into_many(paragraph: str,  max_length=512):
    """ 段落分段 """
    sentences = paragraph.split('。')
    chunks = []
    batch, batch_length = [], 0
    for sentence in sentences:
        if batch_length >= max_length:
            chunks.append("".join(batch))
            batch, batch_length = [], 0
        else:
            sentence = sentence.strip() + "。"
            sentence_length = len(sentence)
            if batch_length + sentence_length <= max_length:
                batch.append(sentence)
                batch_length += sentence_length
            else:
                chunks.append("".join(batch))
                batch, batch_length = [], 0
                if sentence_length > max_length:
                    batch = [sentence[:max_length]
                             for i in range(0, len(sentence), max_length)]
                    chunks.append("".join(batch))
                    batch, batch_length = [], 0
                else:
                    batch.append(sentence)
                    batch_length += sentence_length
    if batch_length:
        chunks.append("".join(batch))
    return chunks


def split_text_into_many(text: str, min_length=1, max_length = 512, max_paragraphs=100):
    """ 文章文段 """

    # 段落
    paragraphs = text.split("\n")
    chunks, chunks_num = [], 0
    batch, batch_length = [], 0
    for paragraph in paragraphs:
        paragraph = paragraph.strip().replace('\xa0', '')
        paragraph = paragraph.replace("\u2003", "")
        if chunks_num >= max_paragraphs:
            chunks.append("".join(batch))
            batch, batch_length = [], 0
        else:
            paragraph_length = len(paragraph)
            # paragraph_length+ batch_length <= max_length
            if (paragraph_length + batch_length) <= max_length:
                batch.append(paragraph)
                batch_length += paragraph_length
            else:
                if batch_length:
                    chunks.append("".join(batch))
                batch, batch_length = [], 0
                if paragraph_length <= max_length:
                    batch.append(paragraph)
                    batch_length += paragraph_length
                else:
                    chunks.extend(split_paragraph_into_many(paragraph, max_length))
                    batch, batch_length = [], 0
    if batch_length:
        chunks.append("".join(batch))
    return chunks


def parse_args():
    parser = argparse.ArgumentParser(description="舆情分析")
    parser.add_argument("--host", type=str, required=False, default="0.0.0.0")
    parser.add_argument("--port", type=int, required=False, default=6006)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--extra_model_name_or_path", type=str, required=False, default="")
    parser.add_argument("--inference_mode", type=str, required=False, default="origin")
    parser.add_argument("--model_type", type=str, required=False, default="ft")
    parser.add_argument("--pre_seq_len", type=int, required=True)
    parser.add_argument("--quantize", type=bool, required=False, default=False)
    args = parser.parse_args()
    return args


def get_word_idx(text, entity):
    start_idx = text.index(entity)
    end_idx = start_idx + len(entity)
    return start_idx, end_idx


def build_doccano_label(text, labels):
    res = []
    for item in labels:
        try:
            label, entity = item.split("-")
            index = get_word_idx(text, entity)
        except:
            continue
        res.append([index[0], index[1], label])
    return res


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    """Return state_dict with weights of LoRA's A and B matrices and with biases depending on the `bias` value.

    Args:
        model: model with LoRA layers
        bias: 
            ``"none"``: state dict will not store bias weights,
            ``"lora_only"``: state dict will store bias weights only from LoRA layers,
            ``"all"``: state dict will store all bias weights.

    Returns:
        Weights and biases of LoRA layers

    Raises:
        NotImplementedError: if `bias` not in ["none", "lora_only", "all"]
    """
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError