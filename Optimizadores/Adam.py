import torch
import numpy as np 

# other libraries
from typing import Iterator, Dict, Any, DefaultDict 
class Adam(torch.optim.Optimizer):
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
        lr=1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
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
                gradiente_p = p.grad 
                if group["weight_decay"] != 0: 
                    gradiente_p = gradiente_p + group["weight_decay"]*p.data
                param_group = self.state[p]
                if ("momentum_buffer1" not in param_group) and ("momentum_buffer2" not in param_group) and ("t" not in param_group): 
                    param_group["momentum_buffer1"] = torch.zeros_like(p.data)
                    param_group["momentum_buffer2"] = torch.zeros_like(p.data)
                    param_group["t"] = 0 
                momentum_1 = param_group["momentum_buffer1"]
                momentum_2 = param_group["momentum_buffer2"]
                t = param_group["t"]

                momentum_1 = group["betas"][0]*momentum_1 + (1 - group["betas"][0])*gradiente_p
                momentum_2 = group["betas"][1]*momentum_2 + (1 - group["betas"][1])*gradiente_p**2
                t += 1 

                momentum_1_hat = momentum_1/(1 - group["betas"][0]**t)
                momentum_2_hat = momentum_2/(1 - group["betas"][1]**t)

                param_group["momentum_buffer1"] = momentum_1
                param_group["momentum_buffer2"] = momentum_2
                param_group["t"] = t 

                p.data = p.data - (group["lr"]*momentum_1_hat)/(torch.sqrt(momentum_2_hat) + group["eps"])