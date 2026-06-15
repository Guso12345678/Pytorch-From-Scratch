import torch

class SilUFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx:any, inputs:torch.Tensor):
        ctx.save_for_backward(inputs)
        outputs = (inputs)/(1 + torch.exp(-inputs))
        return outputs

    @staticmethod
    def backward(ctx:any, grad_outputs:torch.Tensor): 
        inputs, = ctx.saved_tensors
        grad_inputs = grad_outputs.clone()
        grad_inputs *= ((1+ torch.exp(inputs))+(inputs*torch.exp(inputs)))/(1+torch.exp(inputs))**2
        return grad_inputs


class SiLU(torch.nn.Module): 
    def __init__(self): 
        super().__init__()
        self.fn = SilUFunction.apply

    def forward(self, inputs:torch.Tensor): 
        return self.fn(inputs)