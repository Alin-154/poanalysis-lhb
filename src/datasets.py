#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   datasets.py
@Time    :   2023/08/15 16:42:51
@Author  :   nicholas wu 
@Version :   1.0
@Contact :   nicholas_wu@aliyun.com
@License :    
@Desc    :   None
'''
import torch
from torch.utils.data import Dataset
from .utils import read_jsonlines


class ChatGLMDataset(Dataset):
    def __init__(self, path, tokenizer, max_source_length=64, max_target_length=64, \
                 input_column="input", output_column="output", \
                    ignore_pad_token_for_loss=True) -> None:
        super().__init__()
        self.data = read_jsonlines(path)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.input_column = input_column
        self.output_column = output_column
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss

    def collate_fn(self, batch):
        x = []
        y = []
        max_seq_length = self.max_source_length + self.max_target_length
        for i in batch:
            input, output = i[self.input_column], i[self.output_column]
            a_ids = self.tokenizer.encode(text=input, add_special_tokens=False)
            b_ids = self.tokenizer.encode(text=output, add_special_tokens=False)
            if len(a_ids) > self.max_source_length - 1:
                a_ids = a_ids[: self.max_source_length - 1]

            if len(b_ids) > self.max_target_length - 2:
                b_ids = b_ids[: self.max_target_length - 2]

            input_ids = self.tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

            context_length = input_ids.index(self.tokenizer.bos_token_id)
            mask_position = context_length - 1
            labels = [-100] * context_length + input_ids[mask_position+1:]
            
            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [self.tokenizer.pad_token_id] * pad_len
            if self.ignore_pad_token_for_loss:
                labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]
            x.append(input_ids)
            y.append(labels)
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        return x, y

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)