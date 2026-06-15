import torch
import numpy as np 

# other libraries
from typing import Iterator, Dict, Any, DefaultDict 

class RMSPROP(torch.optim.Optimizer):
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
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0, 
        centered:bool = False
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # TODO
        defaults: Dict[Any, Any] = dict(lr=lr, weight_decay=weight_decay,alpha=alpha,eps=eps,momentum=momentum,centered=centered)
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
                if "momentum_buffer1" not in param_group: 
                    param_group["momentum_buffer1"] = torch.zeros_like(p.data)
                velocidad = param_group["momentum_buffer1"]
                velocidad = group["alpha"]*velocidad + (1 - group["alpha"])*gradiente_p**2
                param_group["momentum_buffer1"] = velocidad 
                velocidad_gorro = velocidad 
                if group["centered"] == True: 
                    if "momentum_buffer2" not in param_group: 
                        param_group["momentum_buffer2"] = torch.zeros_like(p.data)
                    g_ave = param_group["momentum_buffer2"]
                    g_ave = g_ave*group["alpha"] + (1 - group["alpha"])*gradiente_p
                    param_group["momentum_buffer2"] = g_ave
                    velocidad_gorro = velocidad_gorro - (g_ave)**2
                if group["momentum"] > 0: 
                    if "momentum_buffer3" not in param_group: 
                        param_group["momentum_buffer3"] = torch.zeros_like(p.data)
                    bt = param_group["momentum_buffer3"]
                    bt = group["momentum"]*bt + (gradiente_p)/(torch.sqrt(velocidad_gorro) + group["eps"])
                    param_group["momentum_buffer3"] = bt 
                    p.data = p.data - group["lr"]*bt
                else: 
                    p.data = p.data - (group["lr"]*gradiente_p)/(torch.sqrt(velocidad_gorro) + group["eps"])