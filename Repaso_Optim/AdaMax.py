import torch
import numpy as np 

# other libraries
from typing import Iterator, Dict, Any, DefaultDict 
class Adamax(torch.optim.Optimizer):
    """
    This class is a custom implementation of the Adam algorithm.

    Attr:
        param_groups: list with the dict of the parameters.
        state: dict with the state for each parameter.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(   
        self,
        params: Iterator[torch.nn.Parameter],
        lr=2e-3,
        betas   : tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # TODO
        defaults: Dict[Any, Any] = dict(lr=lr, weight_decay=weight_decay,betas=betas,eps=eps)
        super().__init__(params,defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """

        # TODO
        for group in self.param_groups: 
            for p in group["params"]: 
                gradiente_p = p.grad.detach()
                if group["weight_decay"] != 0: 
                    gradiente_p = gradiente_p + group["weight_decay"]*p.data 
                
                param_group = self.state[p]
                if ("momentum1" not in param_group) and ("infinity_norm" not in param_group) and ("t" not in param_group): 
                    param_group["momentum1"] = torch.zeros_like(p.data)
                    param_group["infinity_norm"] = torch.zeros_like(p.data) 
                    param_group["t"] = 0 
                momentum1 = param_group["momentum1"]
                infinity_norm = param_group["infinity_norm"]
                t = param_group["t"]
                t = t + 1
                momentum1 = group["betas"][0]*momentum1 + (1 - group["betas"][0])*gradiente_p
                infinity_norm = torch.max(group["betas"][1]*infinity_norm, torch.abs(gradiente_p) + group["eps"])
                param_group["momentum1"] = momentum1
                param_group["infinity_norm"] = infinity_norm
                p.data = p.data - (group["lr"]*momentum1)/((1 - group["betas"][0]**t)*infinity_norm)