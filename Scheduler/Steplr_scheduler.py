import torch
import numpy as np
import torch.nn.functional as F
from torch.jit import RecursiveScriptModule

# other libraries
import os
import random
from typing import Optional

class StepLr(torch.optim.lr_scheduler.LRScheduler): 
    """
    This

    Attr:
        optimizer: optimizer that the scheduler is using.
        step_size: number of steps to decrease learning rate.
        gamma: factor to decrease learning rate.
        count: count of steps.
    """
    def __init__(self,optimizer:torch.optim.Optimizer,step_size:int,gamma:float, last_epoch: int = -1): 
        self.step_size = step_size 
        self.gamma = gamma 
        self.counters = 0
        super().__init__(optimizer,last_epoch)
    
    def step(self, epoch = None):
        if (self.counters % self.step_size == 0) and (self.counters != 0): 
            for param_group in self.optimizer.param_groups: 
                param_group["lr"] = param_group["lr"]*self.gamma
        self.counters +=1 
    
    #Forma de chat y es muy parecido al codigo fuente de la funcion (me gusta menos aunque mas optima): 
    def get_lr(self):
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0: 
            return [group["lr"] for group in self.optimizer.param_groups]
        return [lr*self.gamma for lr in self.get_last_lr()]

        