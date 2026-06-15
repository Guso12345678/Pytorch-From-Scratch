import torch
import numpy as np 

# other libraries
from typing import Iterator, Dict, Any, DefaultDict 
class Adagrad(torch.optim.Optimizer):
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
        lr_decay = 0, 
        initial_acumulator_value = 0, 
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
        defaults: Dict[Any, Any] = dict(lr=lr, weight_decay=weight_decay,lr_decay = lr_decay,initial_acumulator_value=initial_acumulator_value,eps=eps)
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
                gradient_p = p.grad
                param_group = self.state[p]
                if ("state_sum" not in param_group): 
                    param_group["state_sum"] = group["initial_acumulator_value"]
                    param_group["t"] = 0
                t = param_group["t"]
                t += 1 
                state_sum = param_group["state_sum"]
                learning_rate_gorrito = group["lr"]/(1 + (t - 1)*group["lr_decay"])
                if group["weight_decay"] != 0: 
                    gradient_p = gradient_p + group["weight_decay"]*p.data 
                state_sum = state_sum + gradient_p**2 
                param_group["state_sum"] = state_sum
                param_group["t"] = t 
                p.data = p.data - learning_rate_gorrito*((gradient_p)/(torch.sqrt(state_sum) + group["eps"]))

                    