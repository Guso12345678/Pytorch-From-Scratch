import torch

class TanhshrinkFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx:any, inputs:torch.Tensor):
        outputs = inputs.clone()
        ctx.save_for_backward(inputs)
        outputs = inputs - (torch.exp(inputs)-torch.exp(-inputs))/(torch.exp(inputs)+torch.exp(-inputs))
        return outputs

    @staticmethod
    def backward(ctx:any, grad_outputs:torch.Tensor): 
        inputs, = ctx.saved_tensors
        grad_inputs = grad_outputs.clone()
        grad_inputs *= 1 - ((torch.exp(inputs)-torch.exp(-inputs))/(torch.exp(inputs)+torch.exp(-inputs)))**2
        return grad_inputs,None


class Tanhshrink(torch.nn.Module): 
    def __init__(self): 
        super().__init__()
        self.fn = TanhshrinkFunction.apply

    def forward(self, inputs:torch.Tensor): 
        return self.fn(inputs)