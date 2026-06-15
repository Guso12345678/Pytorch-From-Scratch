import torch
import numpy as np
import torch.nn.functional as F
from torch.jit import RecursiveScriptModule

# other libraries
import os
import random
from typing import Optional

class MultiplicativeLR(torch.optim.lr_scheduler.LRScheduler):
    """
    This

    Attr:
        optimizer: optimizer that the scheduler is using.
        step_size: number of steps to decrease learning rate.
        gamma: factor to decrease learning rate.
        count: count of steps.
    """

    optimizer: torch.optim.Optimizer
    last_epoch: int
    counters: int

    def __init__(
        self,
        optimizer: torch.optim.Optimizer, 
        lambda_fn, 
        last_epoch:int = -1
    ) -> None:
        self.lambda_fn = lambda_fn  
        self.counters = 0
        super().__init__(optimizer,last_epoch)
    
    #MI implementacion: 
    def step(self, epoch = None):
        for i, param_group in enumerate(self.optimizer.param_groups): 
            param_group["lr"] = param_group["lr"] * self.lambda_fn(self.counters)
        self.counters += 1
    #Chat, no recomendado modificar el el group["lr"] desde aqui es mas usado para cuando solo se usa el primero , es decir, sin sobreescribir el valor del mismo.  
    def get_lr(self):
        return [
            group["lr"] * self.lambda_fn(self.last_epoch)
            for group in self.optimizer.param_groups
        ]    