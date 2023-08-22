#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   callbacks.py
@Time    :   2023/08/18 09:25:43
@Author  :   nicholas wu 
@Version :   1.0
@Contact :   nicholas_wu@aliyun.com
@License :    
@Desc    :   None
'''
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from weakref import proxy
from .utils import lora_state_dict
from .chatglm_lora import PlChatGLM


class LLMModelCheckpoint(ModelCheckpoint):

    def _save_checkpoint(self, trainer: 'pl.Trainer', filepath: str) -> None:
        if trainer.model.config.use_lora:
            bias = trainer.model.config.lora_config.bias
            lora_model = lora_state_dict(trainer.model, bias)
            torch.save(lora_model, filepath)
        else:
            trainer.save_checkpoint(filepath, self.save_weights_only)

        self._last_global_step_saved = trainer.global_step

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))