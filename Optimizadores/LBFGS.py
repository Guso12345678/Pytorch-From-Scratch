import torch
import numpy as np 

# other libraries
from typing import Iterator, Dict, Any, DefaultDict 
class LBFGS(torch.optim.Optimizer):
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
        lr=1,
        max_iter:int=20,
        eps: float = 1e-7,
        history_size=100
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # TODO
        defaults: Dict[Any, Any] = dict(lr=lr, history_size=history_size,max_iter=max_iter,eps=eps)
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
        loss = closure() 
        for group in self.param_groups: 
            for p in group["params"]:
                param_group = self.state[p] 
                gradiente_p = p.grad.detach().clone()

                norma_gradiente = torch.sqrt(torch.sum(gradiente_p**2))
                if norma_gradiente < group["eps"]: 
                    return loss  
                    
                if ("s_hist" not in param_group) and ("y_hist" not in param_group): 
                    param_group["s_hist"] = []
                    param_group["y_hist"] = []
                s_hist = param_group["s_hist"]
                y_hist = param_group["y_hist"]
                q = gradiente_p.clone()
                alphas = []


                if (len(s_hist) > 0) and (len(y_hist) > 0): 

                    for i in reversed(range(len(s_hist))): 
                        s = s_hist[i]
                        y = y_hist[i]
                        alpha = torch.dot(s.flatten(), q.flatten()) / torch.dot(y.flatten(), s.flatten())
                        q = q - alpha*y 
                        alphas.append(alpha)
                    y = y_hist[-1]
                    s = s_hist[-1]
                    H_k0 = torch.dot(s.flatten(), y.flatten()) / torch.dot(y.flatten(), y.flatten())
                    r = H_k0 * q
                else:
                    r = q

                for i in range(len(s_hist)): 
                    s = s_hist[i]
                    y = y_hist[i] 
                    beta = torch.dot(y.flatten(), r.flatten()) / torch.dot(y.flatten(), s.flatten())
                    alpha = alphas[len(s_hist) - 1 - i]
                    r = r + s * (alpha - beta)
                    
                p.data = p.data - group["lr"] * r
                loss = closure()
                grad_new = p.grad.detach().clone() 

                s_k = (-group["lr"] * r).detach() 
                y_k = (grad_new - gradiente_p).detach()
                if len(s_hist) >= group["history_size"] and len(y_hist) >= group["history_size"]: 
                    s_hist.pop(0)
                    y_hist.pop(0)
                s_hist.append(s_k)
                y_hist.append(y_k)
        return loss 
            


