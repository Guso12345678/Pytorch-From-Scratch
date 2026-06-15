# deep learning libraries
import torch
import numpy as np
import torch.nn.functional as F
from torch.jit import RecursiveScriptModule

# other libraries
import os
import random
from typing import Optional


class LambdaLR(torch.optim.lr_scheduler.LRScheduler):
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
    
    #Esta es la implementacion correcta. 
    def get_lr(self):
        lista = [base_lr * self.lambda_fn(self.last_epoch + 1) for base_lr in self.base_lrs]
        return lista 
    
    #Prueba Mia: 
    def lambda_fn1(self,epoch): 
        return 0.95**epoch
    
    def step(self, epoch = None):
        for i,param_group in enumerate(self.optimizer.param_groups):
            lr_inicial = self.base_lrs[i] 
            param_group["lr"] = lr_inicial * self.lambda_fn1(self.counters)
        self.counters += 1
    