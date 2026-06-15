import torch
import numpy as np 

# other libraries
from typing import Iterator, Dict, Any, DefaultDict 
class RAdam(torch.optim.Optimizer):
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

    def step(self, closure: None = None) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = (-p.grad if group["maximize"] else p.grad).detach()

                if group["weight_decay"] != 0:
                    if group["decoupled_weight_decay"]:
                        p.data = p.data - group["lr"] * group["weight_decay"] * p.data
                    else:
                        grad = grad + group["weight_decay"] * p.data

                state = self.state[p]

                if len(state) == 0:
                    state["momentum1"] = torch.zeros_like(p.data)
                    state["momentum2"] = torch.zeros_like(p.data)
                    state["t"] = 0

                m1, m2, t = state["momentum1"], state["momentum2"], state["t"]

                t += 1
                m1 = group["betas"][0] * m1 + (1 - group["betas"][0]) * grad
                m2 = group["betas"][1] * m2 + (1 - group["betas"][1]) * grad ** 2

                state["momentum1"] = m1
                state["momentum2"] = m2
                state["t"] = t

                m1_hat = m1 / (1 - group["betas"][0] ** t)

                beta2_t = group["betas"][1] ** t
                rho_inf = 2 / (1 - group["betas"][1]) - 1
                rho_t = rho_inf - (2 * t * beta2_t) / (1 - beta2_t)

                if rho_t > 5:
                    rt = torch.sqrt(
                        ((rho_t - 4) * (rho_t - 2) * rho_inf) /
                        ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                    )
                    lt = torch.sqrt(1 - beta2_t) / (torch.sqrt(m2) + group["eps"])
                    p.data = p.data - group["lr"] * m1_hat * rt * lt
                else:
                    p.data = p.data - group["lr"] * m1_hat

                 