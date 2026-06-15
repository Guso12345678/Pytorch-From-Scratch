import torch
from torch.optim import Optimizer
from collections import defaultdict
from typing import Iterator, Dict, Any

class RProp(Optimizer):
    def __init__(self, params: Iterator[torch.nn.Parameter],
                 lr: float = 1e-2,
                 etas=(0.5, 1.2),
                 step_sizes=(1e-6, 50.0)) -> None:

        defaults = dict(lr=lr, etas=etas, step_sizes=step_sizes)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
    def step(self, closure=None): 
        for group in self.param_groups: 
            for p in group["params"]:
                gradiente_p = p.grad
                param_group = self.state[p]
                if ("momentum_gradient" not in param_group) and ("lr" not in param_group): 
                    param_group["momentum_gradient"] = torch.zeros_like(p.data)
                    param_group["lr"] = group["lr"]
                gradiente_previo = param_group["momentum_gradient"]
                lr = param_group["lr"]
                if (gradiente_p * gradiente_previo).gt(0).all(): 
                    lr = min(lr*group["etas"][0],group["step_sizes"][0])
                    param_group["lr"] = lr 
                elif (gradiente_p * gradiente_previo).lt(0).all():
                    lr = max(lr*group["etas"][1],group["step_sizes"][1])
                    param_group["lr"] = lr 
                    gradiente_p = 0 
                else: 
                    lr = lr 
                    param_group["lr"] = lr 
                p.data = p.data - lr*torch.sign(gradiente_p)
                param_group["momentum_gradient"] = gradiente_p