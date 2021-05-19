# coding: utf-8
from typing import List

import torch
import torch.nn as nn


class ModelBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.backup = {}
        self.model: nn.Module = None

    def attack(self, emb_names: List[str], epsilon: float = 1.):
        for name, param in self.model.named_parameters():
            for emb_name in emb_names:
                if param.requires_grad and emb_name in name:
                    self.backup[name] = param.data.clone()
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        r_at = epsilon * param.grad / norm
                        param.data.add_(r_at)

    def restore(self, emb_names: List[str]):
        for name, param in self.model.named_parameters():
            for emb_name in emb_names:
                if param.requires_grad and emb_name in name:
                    assert name in self.backup
                    param.data = self.backup[name]
        self.backup = {}
