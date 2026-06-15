import torch
import numpy as np
import torch.nn.functional as F
from torch.jit import RecursiveScriptModule

# other libraries
import os
import random
from typing import Optional


####IMPLEMENTACION MIA Y FUNCIONA. 
class CyclicLr(torch.optim.lr_scheduler.LRScheduler): 
    """
    This

    Attr:
        optimizer: optimizer that the scheduler is using.
        step_size: number of steps to decrease learning rate.
        gamma: factor to decrease learning rate.
        count: count of steps.
    """
    def __init__(self,optimizer:torch.optim.Optimizer,mode ,base_lr,max_lr,gamma, cycle_momentum, base_momentum,max_momentum,last_epoch: int = -1): 
        self.mode = mode
        self.counters = 0
        self.step_size_up = 2000
        self.step_size_down = 4000
        self.amplitud_inicial = max_lr - base_lr
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.gamma = gamma
        self.numero_ciclos = 0 

        self.cycle_momentum = cycle_momentum
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        super().__init__(optimizer,last_epoch)
    
    def step(self, epoch = None):
        if self.mode == "triangular": 
            if 0 <= self.counters < self.step_size_up:
                for i, param_group in enumerate(self.optimizer.param_groups): 
                    param_group["lr"] = self.base_lr + ((self.counters)/(self.step_size_up))*self.amplitud_inicial
                    if (self.cycle_momentum == True) and ("momentum" in param_group): 
                        param_group["momentum"] = self.max_momentum - (self.counters/self.step_size_up)*(self.max_momentum - self.base_momentum)
            if self.step_size_up <= self.counters < self.step_size_down: 
                for i, param_group in enumerate(self.optimizer.param_groups): 
                    param_group["lr"] = self.max_lr - ((self.counters)/self.step_size_down)*self.amplitud_inicial
                    if (self.cycle_momentum == True) and ("momentum" in param_group): 
                        param_group["momentum"] = self.base_momentum+ (self.counters/self.step_size_down)*(self.max_momentum - self.base_momentum)
            if self.counters == self.step_size_down: 
                self.counters = 0 
                self.numero_ciclos += 1 
        if self.mode == "triangular2": 
            if 0 <= self.counters < self.step_size_up:
                for i, param_group in enumerate(self.optimizer.param_groups): 
                    param_group["lr"] = self.base_lr + ((self.counters)/(self.step_size_up))*self.amplitud_inicial
                    if (self.cycle_momentum == True) and ("momentum" in param_group): 
                        param_group["momentum"] = self.max_momentum - (self.counters/self.step_size_up)*(self.max_momentum - self.base_momentum)
            if self.step_size_up <= self.counters < self.step_size_down: 
                for i, param_group in enumerate(self.optimizer.param_groups): 
                    param_group["lr"] = self.max_lr - ((self.counters)/self.step_size_down)*self.amplitud_inicial
                    if (self.cycle_momentum == True) and ("momentum" in param_group): 
                        param_group["momentum"] = self.base_momentum + (self.counters/self.step_size_down)*(self.max_momentum - self.base_momentum)
            if self.counters == self.step_size_down: 
                self.counters = 0 
                self.numero_ciclos += 1
                self.amplitud_inicial = self.amplitud_inicial / 2   
        if self.mode == "exp_range": 
            if 0 <= self.counters < self.step_size_up:
                for i, param_group in enumerate(self.optimizer.param_groups): 
                    param_group["lr"] = self.base_lr + ((self.counters)/(self.step_size_up))*self.amplitud_inicial
                    if (self.cycle_momentum == True) and ("momentum" in param_group): 
                        param_group["momentum"] = self.max_momentum - (self.counters/self.step_size_up)*(self.max_momentum - self.base_momentum)
            if self.step_size_up <= self.counters < self.step_size_down: 
                for i, param_group in enumerate(self.optimizer.param_groups): 
                    param_group["lr"] = self.max_lr - ((self.counters)/self.step_size_down)*self.amplitud_inicial
                    if (self.cycle_momentum == True) and ("momentum" in param_group): 
                        param_group["momentum"] = self.base_momentum + (self.counters/self.step_size_down)*(self.max_momentum - self.base_momentum)
            if self.counters == self.step_size_down: 
                self.counters = 0 
                self.numero_ciclos += 1
                self.amplitud_inicial = self.amplitud_inicial * self.gamma**self.numero_ciclos 



