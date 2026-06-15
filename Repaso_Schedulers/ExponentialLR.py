import torch
import numpy as np
import torch.nn.functional as F
from torch.jit import RecursiveScriptModule

# other libraries
import os
import random
from typing import Optional

class ExponentialLr(torch.optim.lr_scheduler.LRScheduler): 
    """
    This

    Attr:
        optimizer: optimizer that the scheduler is using.
        step_size: number of steps to decrease learning rate.
        gamma: factor to decrease learning rate.
        count: count of steps.
    """
    def __init__(self,optimizer:torch.optim.Optimizer, gamma:float, last_epoch: int = -1): 
        self.gamma = gamma
        self.counters = 0
        super().__init__(optimizer,last_epoch)
    
    def step(self, epoch = None):
        for i, param_group in enumerate(self.optimizer.param_groups): 
            initial_lr = self.base_lrs[i]
            param_group["lr"] = initial_lr*self.gamma**self.counters 
        
        self.counters += 1 