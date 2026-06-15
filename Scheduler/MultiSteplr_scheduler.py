import torch
import numpy as np
import torch.nn.functional as F
from torch.jit import RecursiveScriptModule

# other libraries
import os
import random
from typing import Optional

class MultiStepLr(torch.optim.lr_scheduler.LRScheduler): 
    """
    This

    Attr:
        optimizer: optimizer that the scheduler is using.
        step_size: number of steps to decrease learning rate.
        gamma: factor to decrease learning rate.
        count: count of steps.
    """
    def __init__(self,optimizer:torch.optim.Optimizer,milestones:list,gamma:float, last_epoch: int = -1): 
        self.milestones = milestones
        self.gamma = gamma 
        self.counters = 0
        super().__init__(optimizer,last_epoch)
    
    #La mia funciona perfectamente pero menos eficiente: 
    def step(self, epoch = None):
        if self.counters in self.milestones: 
            for param_group in self.optimizer.param_groups: 
                param_group["lr"] = param_group["lr"]*self.gamma
        self.counters += 1
    
    #Chat mas eficiente: 
    def get_lr(self):
        if self.last_epoch in self.milestones:
            return [group['lr'] * self.gamma for group in self.optimizer.param_groups]
        return [group['lr'] for group in self.optimizer.param_groups]