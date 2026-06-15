import torch 
import torch.nn as nn 
import numpy as np 

class RReLUFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs,alpha):
        ctx.save_for_backward(inputs)
        ctx.alpha = alpha 
        mask1 = inputs >= 0
        mask2 = inputs < 0 
        outputs = torch.zeros_like(inputs)
        outputs[mask1] = inputs[mask1]
        outputs[mask2] = alpha*inputs[mask2]
        return outputs 
    @staticmethod
    def backward(ctx,grad_outputs): 
        inputs, = ctx.saved_tensors
        alpha = ctx.alpha 
        mask1 = inputs >= 0 
        mask2 = inputs < 0 
        grad_inputs = torch.zeros_like(grad_outputs) 
        grad_inputs[mask1] = 1 
        grad_inputs[mask2] = alpha 

        return grad_inputs*grad_outputs, None, None 


class RReLU(nn.Module):
    def __init__(self, lower=1.0 / 8, upper=1.0 / 3):
        super().__init__()
        self.lower = lower
        self.upper = upper

    def forward(self, x):
        if self.training:
            alpha = torch.empty(1, device=x.device).uniform_(self.lower, self.upper).item()
        else:
            alpha = (self.lower + self.upper) / 2.0
        return RReLUFunction.apply(x, alpha) 