import torch
import numpy as np
import torch.nn.functional as F
from torch.jit import RecursiveScriptModule

# other libraries
import os
import random
from typing import Optional

class CosinneAnnealingLr(torch.optim.lr_scheduler.LRScheduler): 
    """
    This

    Attr:
        optimizer: optimizer that the scheduler is using.
        step_size: number of steps to decrease learning rate.
        gamma: factor to decrease learning rate.
        count: count of steps.
    """
    def __init__(self,optimizer:torch.optim.Optimizer, T_0:int, T_mult:int,eta_min:float,last_epoch: int = -1): 
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_0 = T_0
        self.T_mult = T_mult 
        super().__init__(optimizer,last_epoch)
    def step(self, epoch = None):
        if self.T_cur == 0: 
            for i, param_group in enumerate(self.optimizer.param_groups): 
                param_group["lr"] = self.base_lrs[i]
        if 0 < self.T_cur < self.T_0: 
            for i,param_group in enumerate(self.optimizer.param_groups): 
                param_group["lr"] = self.eta_min + 0.5*(self.base_lrs[i] - self.eta_min)*(1 + np.cos(((self.T_cur)/(self.T_0))*np.pi))
        if self.T_cur == self.T_0: 
            for i, param_group in enumerate(self.param_groups): 
                param_group["lr"] = self.eta_min
            self.T_cur = -1 
            self.T_0 = self.T_0*self.T_mult
        self.T_cur += 1  
