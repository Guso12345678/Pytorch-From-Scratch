import torch

class SoftPlusFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx:any, inputs:torch.Tensor, beta:float):
        outputs = inputs.clone()
        ctx.save_for_backward(inputs)
        ctx.beta = beta 
        outputs = (1/beta) * torch.log(1 + torch.exp(beta*inputs))
        return outputs

    @staticmethod
    def backward(ctx:any, grad_outputs:torch.Tensor): 
        inputs, = ctx.saved_tensors
        beta = ctx.beta
        grad_inputs = grad_outputs.clone()
        grad_inputs *= (beta * torch.exp(beta*inputs)) / (1+torch.exp(beta*inputs))
        return grad_inputs


class SoftPlus(torch.nn.Module): 
    def __init__(self, beta:float=1.0): 
        super().__init__()
        self.fn = SoftPlusFunction.apply
        self.beta = beta 

    def forward(self, inputs:torch.Tensor): 
        return self.fn(inputs,self.beta)