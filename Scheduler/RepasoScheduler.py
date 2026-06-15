import torch
import numpy as np
import torch.nn.functional as F
from torch.jit import RecursiveScriptModule
import math 

# other libraries
import os
import random
from typing import Optional


#####PRIMERO LA LINEAL: 
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
            for i, param_group in enumerate(self.optimizer.param_groups): 
                initial_lr = self.base_lrs[i]
                param_group["lr"] = initial_lr*((1 - (self.counters/self.total_iters))*self.start_factor + (self.counters/self.total_iters)*self.end_factor)
        else: 
            for i, param_group in enumerate(self.optimizer.param_groups): 
                initial_lr = self.base_lrs[i]
                param_group["lr"] = initial_lr*self.end_factor


#####EXPONENTIAL LR:

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


#####CosinneAnnealingLR: 
class CosinneAnnealingLr(torch.optim.lr_scheduler.LRScheduler): 

    def __init__(self,optimizer:torch.Tensor,T_max:int,eta_min:float,last_epoch: int = -1): 
        self.eta_min = eta_min 
        self.T_max = T_max 
        self.T_cur = 0 
        super().__init__(optimizer,last_epoch)
    def step(self,epoch = None): 
        for i, param_group in enumerate(self.optimizer.param_groups): 
            eta_max = self.base_lrs[i]
            param_group["lr"] = self.eta_min + 0.5*(eta_max - self.eta_min)*(1 + torch.cos((self.T_cur)/(self.T_max)*torch.pi).item())
        self.T_cur += 1 


#####ReduceLROnPlateauLR: 
class ReduceLROnPlateauLr(torch.optim.lr_scheduler.LRScheduler): 
    def __init__(self, optimizer,patience, factor,threshold,last_epoch=-1):
        #self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.best_metric = float("inf")
        self.counters = 0 
        self.epochs_no_improvement = 0 
        super().__init__(optimizer, last_epoch) 
    
    def step(self,metrics): 
        if (abs(self.best_metric - metrics) > self.threshold) and (metrics < self.best_metric): 
            self.best_metric = metrics 
            self.epochs_no_improvement = 0 
        else: 
            self.epochs_no_improvement += 1 
            for i, param_group in enumerate(self.optimizer.param_groups): 
                if self.epochs_no_improvement >= self.patience: 
                    param_group["lr"] = param_group["lr"]*self.factor 
                    self.epochs_no_improvement = 0 
        self.counters += 1 



######CosinneAnnealingWarmRestarts 
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
    
    def step(self,epoch=None): 
        
        if self.T_cur == 0: 
            for i, param_group in enumerate(self.optimizer.param_groups): 
                param_group["lr"] = self.base_lrs[i]
        
        elif 0 < self.T_cur < self.T_0: 
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.eta_min + 0.5*(self.base_lrs[i] - self.eta_min)*(1 + math.cos((self.T_cur/self.T_0)*math.pi))

        elif self.T_cur == 0: 
            for i, param_group in enumerate(self.optimizer.param_groups): 
                param_group["lr"] = self.eta_min
            self.T_cur = -1 
            self.T_0 = self.T_0*self.T_mult

        self.T_cur += 1  




class CyclicLR(torch.optim.lr_scheduler.LRScheduler): 

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
                    param_group["lr"] = self.base_lr + (self.counters/self.step_size_up)*self.amplitud_inicial
                    if (self.cycle_momentum == True) and ("momentum" in param_group): 
                        param_group["momentum"] = self.max_momentum - (self.counters/self.step_size_up)*(self.max_momentum - self.base_momentum)
            elif self.step_size_up <= self.counters < self.step_size_down:
                for i, param_group in enumerate(self.optimizer.param_groups): 
                    param_group["lr"] = self.max_lr - (self.counters/self.step_size_down)*self.amplitud_inicial 
                    if (self.cycle_momentum == True) and ("momentum" in param_group): 
                        param_group["momentum"] = self.base_momentum + (self.counters/self.step_size_down)*(self.max_lr - self.base_lr)
            elif self.counters == self.step_size_down: 
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



        
