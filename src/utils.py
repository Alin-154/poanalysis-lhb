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
from tqdm import tqdm
import jsonlines


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
        if chunks_num >= max_paragraphs:
            chunks.append("".join(batch))
            batch, batch_length = [], 0
        else:
            paragraph = paragraph.strip().replace('\xa0', '')
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
    parser.add_argument("--ptuning_checkpoint", type=str, required=True)
    parser.add_argument("--pre_seq_len", type=int, required=True)
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