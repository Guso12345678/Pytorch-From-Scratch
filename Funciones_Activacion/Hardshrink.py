import torch

class HardSwishFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx:any, inputs:torch.Tensor,alpha:float):
        ctx.save_for_backward(inputs) 
        ctx.alpha = alpha 
        outputs = inputs.clone()
        outputs[(inputs >= -alpha) & (inputs <= alpha) ] = 0 
        return outputs

    @staticmethod
    def backward(ctx:any, grad_outputs:torch.Tensor): 
        inputs, = ctx.saved_tensors
        alpha = ctx.alpha
        grad_inputs = grad_outputs.clone()
        grad_inputs[(inputs > -alpha) & (inputs < alpha)] *= 0

        return grad_inputs, None

class HardSwish(torch.nn.Module): 
    def __init__(self,alpha:float=0.5): 
        super().__init__()
        self.fn = HardSwishFunction.apply
        self.alpha = alpha 

    def forward(self, inputs:torch.Tensor): 
        return self.fn(inputs,self.alpha)