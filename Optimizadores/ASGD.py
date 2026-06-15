import torch
import numpy as np 

# other libraries
from typing import Iterator, Dict, Any, DefaultDict 
class ASGD(torch.optim.Optimizer):
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
        lambd = 0.0001,
        alpha = 0.75, 
        t0 = 100000,
        weight_decay: float = 0.0,
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # TODO
        defaults: Dict[Any, Any] = dict(lr=lr, weight_decay=weight_decay,lambd=lambd,alpha=alpha,t0=t0)
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
                param_group = self.state[p]
                if ("t" not in param_group) and ("n_avg" not in param_group) and ("average_param" not in param_group): 
                    param_group["t"] = 0 
                    param_group["n_avg"] = 0 
                    param_group["average_param"] = p.data.clone()
                t = param_group["t"]
                n_avg = param_group["n_avg"] 
                average_param = param_group["average_param"]
                t += 1  
                if group["weight_decay"]: 
                    gradiente_p = gradiente_p + group["weight_decay"]*p.data
                p.data = p.data - group["lr"]*gradiente_p 
                param_group["t"] = t 
                if t >= group["t0"]: 
                    n_avg = n_avg + 1
                    average_param = average_param + (p.data - average_param)/n_avg 
                    param_group["n_avg"] = n_avg 
                    param_group["average_param"] = average_param
                
