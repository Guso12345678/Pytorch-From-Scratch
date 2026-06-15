import torch
import numpy as np 

# other libraries
from typing import Iterator, Dict, Any, DefaultDict 
class AdaDelta(torch.optim.Optimizer):
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
        lr=1e-2,
        eps = 1e-6,  
        weight_decay: float = 0.0,
        rho = 0.9
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # TODO
        defaults: Dict[Any, Any] = dict(lr=lr, weight_decay=weight_decay,rho=rho,eps=eps)
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
                gradiente_p = p .grad
                if group["weight_decay"] != 0: 
                    gradiente_p = gradiente_p + group["weight_decay"]*p.data
                param_group = self.state[p] 
                if ("momentum_buffer1" not in param_group) and ("momentum_buffer2" not in param_group): 
                    param_group["momentum_buffer1"] = torch.zeros_like(p.data)
                    param_group["momentum_buffer2"] = torch.zeros_like(p.data)
                vt = param_group["momentum_buffer1"]
                ut = param_group["momentum_buffer2"]
                vt = vt*group["rho"] + (gradiente_p**2)*(1 - group["rho"])
                delta_x = ((torch.sqrt(ut + group["eps"]))/(torch.sqrt(vt + group["eps"]))) * gradiente_p
                ut = ut*group["rho"] + (delta_x**2)*(1 - group["rho"])
                param_group["momentum_buffer1"] = vt 
                param_group["momentum_buffer2"] = ut 
                p.data = p.data - group["lr"]*delta_x
