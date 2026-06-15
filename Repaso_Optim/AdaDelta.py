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
        for group in self.param_groups: 
            for p in group["params"]: 
                gradiente_p = p.grad.detach()
                if group["weight_decay"] != 0: 
                    gradiente_p = gradiente_p + group["weight_decay"]*p.data 
                param_group = self.state[p]
                if ("square_avg" not in param_group) and ("accumulative_variables" not in param_group): 
                    param_group["square_avg"] = torch.zeros_like(p.data)
                    param_group["accumulative_variables"] = torch.zeros_like(p.data)
                square_avg = param_group["square_avg"]
                accumulative_variables = param_group["accumulative_variables"]

                square_avg = square_avg*group["rho"] + (gradiente_p**2)*(1 - group["rho"])
                delta_x = (torch.sqrt(accumulative_variables + group["eps"])/(torch.sqrt(square_avg + group["eps"])))*gradiente_p
                accumulative_variables = accumulative_variables*group["rho"] + (delta_x**2)*(1 - group["rho"])
                param_group["square_avg"] = square_avg 
                param_group["accumulative_variables"] = accumulative_variables
                p.data = p.data - group["lr"]*delta_x