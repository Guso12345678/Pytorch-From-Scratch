import torch
import numpy as np 

# other libraries
from typing import Iterator, Dict, Any, DefaultDict 
class AdaFactor(torch.optim.Optimizer):
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
        eps = (None,0.001), 
        beta_decay = -0.8, 
        weight_decay: float = 0.0,
        clipping_factor = 1.0, 
        maximize = False
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # TODO
        defaults: Dict[Any, Any] = dict(lr=lr, weight_decay=weight_decay,beta_decay=beta_decay,clipping_factor = clipping_factor,maximize = maximize,eps=eps)
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
                if group["maximize"] == True: 
                    gradiente_p = - p.grad
                else: 
                    gradiente_p = p.grad 
                param_group = self.state[p]
                if ("momentum_2_row" not in param_group) and ("momentum_2_col" not in param_group) and ("momentum_2_vectors" not in param_group) and ("t" not in param_group):
                    param_group["momentum_2_row"] = torch.zeros(p.data.shape[0], device=p.data.device)
                    param_group["momentum_2_col"] = torch.zeros(p.data.shape[1], device=p.data.device)
                    param_group["momentum_2_vectors"] = torch.zeros_like(p.data)
                    param_group["t"] = 0 
                
                momentum_2_row = param_group["momentum_2_row"]
                momentum_2_col = param_group["momentum_2_col"]
                momentum_2_vectors = param_group["momentum_2_vectors"]
                t = param_group["t"]
                
                t += 1 

                beta_2_pico = 1 - t**group["beta_decay"] 
                rho_t = min(group["lr"],1/np.sqrt(t))
                alpha_t = max(group["eps"][1], torch.sqrt(torch.sum(p.data**2)))*rho_t
                p.data = p.data - group["lr"]*group["weight_decay"]*p.data 

                if len(gradiente_p) > 1: 
                    ones_col = torch.ones(p.data.shape[1], device=p.data.device)
                    momentum_2_row = beta_2_pico * momentum_2_row + (1 - beta_2_pico) * torch.matmul((gradiente_p * gradiente_p), ones_col)
                    ones_row = torch.ones(p.data.shape[0], device=p.data.device)
                    momentum_2_col = beta_2_pico * momentum_2_col + (1 - beta_2_pico) * torch.matmul(ones_row, (gradiente_p * gradiente_p))
                    momentum_2_vectors = (torch.outer(momentum_2_row,momentum_2_col))/torch.max(torch.sum(momentum_2_row),group["eps"][0])
                    param_group["momentum_2_row"] = momentum_2_row
                    param_group["momentum_2_col"] = momentum_2_col
                    param_group["momentum_2_vectors"] = momentum_2_vectors
                else: 
                    momentum_2_vectors = beta_2_pico*momentum_2_vectors + (1 - beta_2_pico)*(gradiente_p*gradiente_p)

                ut = momentum_2_vectors/torch.max(torch.sqrt(momentum_2_vectors),group["eps"][0])
                ut_gorro = ut / torch.max(torch.tensor(1.0, device=ut.device), torch.sqrt(torch.sum(ut)) / group["clipping_factor"])
                p.data = p.data - alpha_t*ut_gorro

                


