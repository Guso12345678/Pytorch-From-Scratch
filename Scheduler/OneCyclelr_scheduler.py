import torch
import numpy as np
import torch.nn.functional as F
from torch.jit import RecursiveScriptModule

# other libraries
import os
import random
from typing import Optional


####IMPLEMENTACION MIA Y FUNCIONA. 
class OneCycleLr(torch.optim.lr_scheduler.LRScheduler): 
    """
    This

    Attr:
        optimizer: optimizer that the scheduler is using.
        step_size: number of steps to decrease learning rate.
        gamma: factor to decrease learning rate.
        count: count of steps.
    """
    def __init__(self,optimizer:torch.optim.Optimizer,pct_start,total_steps,max_lr,anneal_strategy, div_factor,finaldiv_factor,last_epoch: int = -1): 
        self.counters = 0
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.step_size_up = int(total_steps*pct_start) 
        self.step_size_down = int(total_steps - (total_steps*pct_start))
        self.anneal_stategy = anneal_strategy
        self.max_lr = max_lr
        self.div_factor = div_factor
        self.finaldiv_factor = finaldiv_factor
        super().__init__(optimizer,last_epoch)
    def step(self, epoch = None):
        if self.anneal_stategy == "linear": 
            #Fase de subida: 
            if 0 <= self.counters <= self.step_size_up:
                for i,param_group in enumerate(self.optimizer.param_groups): 
                    param_group["lr"] = (self.max_lr/self.div_factor) + (self.max_lr - (self.max_lr/self.div_factor))*(self.counters/self.step_size_up)
            if self.step_size_up < self.counters <= self.step_size_down:
                for i,param_group in enumerate(self.optimizer.param_groups): 
                    param_group["lr"] = self.max_lr - ((self.counters - self.step_size_up)/self.step_size_down)*(self.max_lr - self.max_lr/(self.div_factor*self.finaldiv_factor))
            self.counters += 1  
        elif self.anneal_stategy == "cos": 
            if 0 <= self.counters <= self.step_size_up:
                for i,param_group in enumerate(self.optimizer.param_groups): 
                    param_group["lr"] = (self.max_lr/self.div_factor) + 0.5*(self.max_lr - (self.max_lr/self.div_factor))*(1 - np.cos(np.pi*(self.counters/self.step_size_up)))
            if self.step_size_up < self.counters <= self.total_steps:
                for i,param_group in enumerate(self.optimizer.param_groups): 
                    param_group["lr"] = (self.max_lr)/(self.div_factor*self.finaldiv_factor) + 0.5*(self.max_lr - ((self.max_lr)/(self.div_factor*self.finaldiv_factor)))*(1 + np.cos(np.pi*((self.counters - self.step_size_up)/self.step_size_down)))
            self.counters += 1  
        