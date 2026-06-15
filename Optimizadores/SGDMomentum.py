import torch
import numpy as np 

# other libraries
from typing import Iterator, Dict, Any, DefaultDict

class SGDMomentum(torch.optim.Optimizer):
    """
    This class is a custom implementation of the SGD algorithm with
    momentum.

    Attr:
        param_groups: list with the dict of the parameters.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr=1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # TODO
        defaults: Dict[Any, Any] = dict(lr=lr, weight_decay=weight_decay,momentum=momentum,dampening=0)
        super().__init__(params, defaults) 


    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Attr:
            param_groups: list with the dict of the parameters.
            state: dict with the state for each parameter.
        """

        # TODO
        for group in self.param_groups: 
            for p in group["params"]: 
                gradient_p = p.grad 
                if group["weight_decay"] != 0: 
                    gradient_p = gradient_p + group["weight_decay"]*p.data
                if group["momentum"] != 0: 
                    param_group = self.state[p]
                    if "momentum_buffer" not in param_group: 
                        param_group["momentum_buffer"] = torch.zeros_like(p.data)
                    velocidad = param_group["momentum_buffer"]
                    velocidad = group["momentum"]*velocidad + gradient_p
                    param_group["momentum_buffer"] = velocidad
                else: 
                    velocidad = gradient_p
                p.data = p.data - group["lr"]*velocidad 