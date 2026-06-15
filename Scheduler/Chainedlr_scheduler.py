import torch
import numpy as np
import torch.nn.functional as F
from torch.jit import RecursiveScriptModule

# other libraries
import os
import random
from typing import Optional

class ChainedScheduler(torch.optim.lr_scheduler.LRScheduler):
    """
    Custom implementation of ChainedScheduler. Applies multiple schedulers in sequence,
    each taking over once the previous one finishes.

    Args:
        schedulers (list): List of scheduler instances (custom or PyTorch-based).
        optimizer (Optimizer): The optimizer to update.
    """

    def __init__(self, schedulers, optimizer, milestones,last_epoch=-1):
        self.schedulers = schedulers
        self.optimizer = optimizer 
        self.milestones = milestones
        self.duraciones_scheduler = self.calcular_duraciones_scheduler()
        self.indice_actual = 0
        self.counter = 0 
        super().__init__(self.optimizer, last_epoch)

    def calcular_duraciones_scheduler(self):
        """
        Computes how long each scheduler lasts (based on total_iters or T_max if available).
        Returns a list of (start, end) epochs for each scheduler.
        """
        duraciones = []
        current = 0
        for sched in self.schedulers:
            if hasattr(sched, "total_iters"):
                duracion = sched.total_iters
            elif hasattr(sched, "T_max"):
                duracion = sched.T_max
            else:
                duracion = float("inf")

            duraciones.append((current, current + duracion))
            current += duracion
        return duraciones

    def step(self, epoch=None):
        # Incrementa el epoch automáticamente

        for i,(start,end) in enumerate(self.duraciones_scheduler): 
            if start <= self.counter < end: 
                self.indice_actual = i 
                break 
            if self.counter >= self.duraciones_scheduler[-1][1]: 
                self.indice_actual = len(self.duraciones_scheduler) -1 
                break 
        self.schedulers[self.indice_actual].step()
        self.counter += 1
