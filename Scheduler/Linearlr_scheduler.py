import torch
import numpy as np
import torch.nn.functional as F
from torch.jit import RecursiveScriptModule

# other libraries
import os
import random
from typing import Optional

class LinearLr(torch.optim.lr_scheduler.LRScheduler): 
    """
    This

    Attr:
        optimizer: optimizer that the scheduler is using.
        step_size: number of steps to decrease learning rate.
        gamma: factor to decrease learning rate.
        count: count of steps.
    """
    def __init__(self,optimizer:torch.optim.Optimizer, start_factor:float, end_factor:float, total_iters:int, last_epoch: int = -1): 
        self.start_factor = start_factor 
        self.end_factor = end_factor
        self.total_iters = total_iters 
        self.counters = 0
        super().__init__(optimizer,last_epoch)

    def step(self, epoch = None):
        if self.counters <= self.total_iters: 
            for i,param_group in enumerate(self.optimizer.param_groups):
                initial_lr = self.base_lrs[i]
                param_group["lr"] = initial_lr*(((1 - (self.counters/self.total_iters))*self.start_factor) + (self.counters/self.total_iters)*self.end_factor)
        else: 
            for i,param_group in enumerate(self.optimizer.param_groups): 
                initial_lr = self.base_lrs[i]
                param_group["lr"] = initial_lr*self.end_factor
        self.counters += 1 
