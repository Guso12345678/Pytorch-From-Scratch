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
        decoupled_weight_decay:bool = False, 
        maximize:bool = False
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # TODO
        defaults: Dict[Any, Any] = dict(lr=lr, weight_decay=weight_decay,betas=betas,eps=eps,decoupled_weight_decay=decoupled_weight_decay,maximize=maximize)
        super().__init__(params,defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """
        #rho_infinito = 2 / ((1 - self.param_groups[:]["betas"][1]) - 1)
        for group in self.param_groups: 
            for p in group["params"]: 
                rho_infinito = 2 / (1 - group["betas"][1]) - 1
                if group["maximize"] == True: 
                    gradiente_p = - p.grad 
                else: 
                    gradiente_p = p.grad
                param_group = self.state[p]
                if ("momentum_buffer1" not in param_group) and ("momentum_buffer2" not in param_group) and ("t" not in param_group): 
                    param_group["momentum_buffer1"] = torch.zeros_like(p.data)
                    param_group["momentum_buffer2"] = torch.zeros_like(p.data)
                    param_group["t"] = 0 
                momentum_1 = param_group["momentum_buffer1"]
                momentum_2 = param_group["momentum_buffer2"]
                t = param_group["t"]
                if group["weight_decay"] != 0: 
                    if group["decoupled_weight_decay"] == True: 
                        p.data = p.data - group["lr"]*group["weight_decay"]*p.data
                    else: 
                        gradiente_p = gradiente_p + group["weight_decay"]*p.data 
                momentum_1 = (group["betas"][0]*momentum_1) + (1 - group["betas"][0])*gradiente_p
                momentum_2 = (group["betas"][1]*momentum_2) + (1 - group["betas"][1])*gradiente_p**2
                t += 1 

                momentum_1_hat = momentum_1/(1 - group["betas"][0]**t)
                rho_t = rho_infinito - (2*t*group["betas"][1]**t)/(1 - group["betas"][1]**t)

                param_group["momentum_buffer1"] = momentum_1
                param_group["momentum_buffer2"] = momentum_2
                param_group["t"] = t 

                if rho_t > 5: 
                    lt = np.sqrt(1 - group["betas"][1]**t)/(torch.sqrt(momentum_2) + group["eps"])
                    rt = np.sqrt(((rho_t - 4)*(rho_t - 2)*rho_infinito)/((rho_infinito - 4)*(rho_infinito - 2)*rho_t))
                    p.data = p.data - group["lr"]*momentum_1_hat*rt*lt
                else: 
                    p.data = p.data - group["lr"]*momentum_1_hat

