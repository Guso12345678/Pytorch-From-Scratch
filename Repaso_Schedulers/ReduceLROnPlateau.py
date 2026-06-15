import torch
import numpy as np
import torch.nn.functional as F
from torch.jit import RecursiveScriptModule

# other libraries
import os
import random
from typing import Optional

class ReduceLRONPlateauScheduler(torch.optim.lr_scheduler.LRScheduler):
    """
    Custom implementation of ChainedScheduler. Applies multiple schedulers in sequence,
    each taking over once the previous one finishes.

    Args:
        schedulers (list): List of scheduler instances (custom or PyTorch-based).
        optimizer (Optimizer): The optimizer to update.
    """

    def __init__(self, optimizer,patience, factor,threshold,last_epoch=-1):
        # self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.best_metric = float("inf")
        self.counters = 0 
        self.epochs_no_improvement = 0 
        super().__init__(optimizer, last_epoch) 
    
    def step(self,metrics,closure=None): 
        if self.best_metric - metrics >= self.threshold:
            self.epochs_no_improvement = 0 
            self.best_metric = metrics  
        else: 
            self.epochs_no_improvement += 1 
            if self.epochs_no_improvement >= self.patience:
                for i, param_group in enumerate(self.optimizer.param_groups):  
                    param_group["lr"] = param_group["lr"]*self.factor
                    self.epochs_no_improvement = 0  
        self.counters += 1 
            