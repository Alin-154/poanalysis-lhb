#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2023/08/15 17:29:15
@Author  :   nicholas wu 
@Version :   1.0
@Contact :   nicholas_wu@aliyun.com
@License :    
@Desc    :   None
'''

import argparse
from typing import Union, List, Iterable
import pytorch_lightning as pl
from pytorch_lightning.loggers import Logger
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from peft import LoraConfig, TaskType
from src.chatglm_lora import PlChatGLM
from src.datasets import ChatGLMDataset
from src.utils import print_trainable_parameters
from src.callbacks import LLMModelCheckpoint

def train(args):
    use_lora = args.use_lora
    lora_config = None
    if use_lora:
        lora_config = LoraConfig(
                r=args.r,
                lora_alpha=args.lora_alpha,
                target_modules=["query_key_value"],
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False
            )
    config = AutoConfig.from_pretrained(args.model_name_or_path, \
                                        trust_remote_code=True)
    config.model_name_or_path = args.model_name_or_path
    config.use_lora = args.use_lora
    config.lora_config = lora_config
    
    # data & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    train_dataset = ChatGLMDataset(args.train_data_path, tokenizer, \
                                args.max_source_length, \
                                    args.max_target_length, \
                                        args.input_column, \
                                        args.output_column)
    
    train_dataloader = DataLoader(train_dataset, \
                                  batch_size=args.batch_size, \
                                    shuffle=True, \
                                        collate_fn=train_dataset.collate_fn)
    
    # model
    model = PlChatGLM(config)
    print_trainable_parameters(model)


    # train
    model_checkpoint = LLMModelCheckpoint(every_n_epochs=1)
    trainer = pl.Trainer(accelerator=args.accelerator, \
                         max_epochs=args.max_epochs, \
                        logger=args.logger, \
                        precision=args.precision, \
                        callbacks=[model_checkpoint])

    trainer.fit(model, train_dataloaders=train_dataloader)

def parse_args():
    parser = argparse.ArgumentParser(description="chatglm微调")

    # trainer
    parser.add_argument("--accelerator", type=str, required=False, default="auto")
    parser.add_argument("--devices", type=str, required=False, default="auto")
    parser.add_argument("--max_epochs", type=int, required=False, default=1)
    parser.add_argument("--batch_size", type=int, required=False, default=1)
    parser.add_argument("--accumulate_grad_batches", type=int, required=False, default=1)
    parser.add_argument("--precision", type=str, required=False, default="32")
    parser.add_argument("--logger", type=bool, required=False, default=False)

    # model
    parser.add_argument("--model_name_or_path", type=str, required=True)

    # data & tokenizer
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--max_source_length", type=int, required=False, default=256)
    parser.add_argument("--max_target_length", type=int, required=False, default=256)
    parser.add_argument("--input_column", type=str, required=False, default="input")
    parser.add_argument("--output_column", type=str, required=False, default="output")


    # lora
    parser.add_argument("--use_lora", type=bool, required=False, default=True)
    parser.add_argument("--r", type=int, required=False, default=8)
    parser.add_argument("--lora_bias", type=str, required=False, default="none")
    parser.add_argument("--lora_alpha", type=int, required=False, default=32)
    parser.add_argument("--lora_dropout", type=float, required=False, default=0.1)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # train(args)