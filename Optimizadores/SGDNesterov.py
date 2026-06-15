import torch
import numpy as np 

# other libraries
from typing import Iterator, Dict, Any, DefaultDict 

class SGDNesterov(torch.optim.Optimizer):
    """
    This class is a custom implementation of the SGD algorithm with
    momentum.

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

        Args:
            closure: Ignore this parameter. Defaults to None.
        """

        # TODO
        for group in self.param_groups: 
            for p in group["params"]: 
                gradiente_p = p.grad

                if group["weight_decay"] != 0: 
                    gradiente_p = gradiente_p + group["weight_decay"]*p.data

                if group["momentum"] != 0: 
                    param_group = self.state[p]

                    if "momentum_buffer" not in param_group: 
                        param_group["momentum_buffer"] = torch.zeros_like(p.data)
                    
                    velocidad = param_group["momentum_buffer"]
                    velocidad = group["momentum"]*velocidad + gradiente_p
                    param_group["momentum_buffer"] = velocidad 
                
                gradiente_p = gradiente_p + group["momentum"]*velocidad #Solo si hay momentum sino cambiara un poco pero tampoco mucho.
                p.data = p.data - group["lr"]*gradiente_p