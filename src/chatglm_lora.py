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
from peft import PeftModel
from transformers import AutoModel, AutoConfig


class ChatGLM(nn.Module):
    def __init__(self, config: AutoConfig):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(config.model_name_or_path, config=config, trust_remote_code=True)
        if config.use_lora:
            self.base_model = PeftModel(self.base_model, config.lora_config)

    def forward(self, input_ids: Optional[torch.Tensor]):
        lm_logits = self.base_model(input_ids).logits
        return lm_logits


class PlChatGLM(pl.LightningModule):
    def __init__(self, config: AutoConfig) -> None:
        super().__init__()
        self.model = ChatGLM(config)
        self.config = config
        self.outputs = []

    def loss_fct(self, lm_logits, labels):
        lm_logits = lm_logits.to(torch.float32)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        self.outputs.append({"loss": loss})
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.00001, weight_decay=0.0)
        return optimizer

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        lm_logits = self.model(input_ids=input_ids)
        loss = self.loss_fct(lm_logits, labels)
        return loss
    
    def on_train_epoch_end(self) -> None:
        loss = torch.stack([x["loss"] for x in self.outputs]).mean()
        self.log_dict({"avg_loss": loss}, prog_bar=True)
        self.outputs = []
    
    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        lm_logits = self.model(input_ids=input_ids)
        loss = self.loss_fct(lm_logits, labels)
        return loss
