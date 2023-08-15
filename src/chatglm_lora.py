#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   chatglm_lora.py
@Time    :   2023/07/19 15:09:44
@Author  :   nicholas wu 
@Version :   1.0
@Contact :   nicholas_wu@aliyun.com
@License :    
@Desc    :   None
'''
from typing import Optional
import torch
from torch.nn import CrossEntropyLoss
from torch import nn
import pytorch_lightning as pl
from peft import LoraModel, LoraConfig, TaskType
from transformers import AutoModel, AutoConfig
from .utils import print_trainable_parameters


class ChatGLM(nn.Module):
    def __init__(self, config: AutoConfig):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(config.model_name_or_path, config=config, trust_remote_code=True)
        if config.use_lora:
            lora_config = {"default": config.lora_config}
            self.base_model = LoraModel(self.base_model, lora_config, "default")

    def forward(self, input_ids: Optional[torch.Tensor]):
        lm_logits = self.base_model(input_ids)
        return lm_logits


class PlChatGLM(pl.LightningModule):
    def __init__(self, config: AutoConfig) -> None:
        super().__init__(config)
        self.model = ChatGLM(config)
        print_trainable_parameters(self.model)

    def loss_fct(self, lm_logits, labels):
        lm_logits = lm_logits.to(torch.float32)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        lm_logits = self.model(input_ids=input_ids)
        loss = self.loss_fct(lm_logits, labels)
        return loss
    
    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log_dict({"avg_loss": loss}, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        lm_logits = self.model(input_ids=input_ids)
        loss = self.loss_fct(lm_logits, labels)
        return loss
