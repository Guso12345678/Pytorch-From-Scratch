import torch

class SoftSignFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx:any, inputs:torch.Tensor):
        ctx.save_for_backward(inputs)
        outputs = inputs.clone()
        outputs = inputs / (1 + torch.abs(inputs))
        return outputs
    @staticmethod
    def backward(ctx:any, grad_outputs:torch.Tensor): 
        inputs, = ctx.saved_tensors
        grad_inputs = grad_outputs.clone()
        grad_inputs[inputs >= 0] *= (1)/((1 + inputs[inputs >= 0])**2)
        grad_inputs[inputs < 0] *= (1)/((1 - inputs[inputs < 0]))**2
        return grad_inputs

class SoftSign(torch.nn.Module): 
    def __init__(self): 
        super().__init__()
        self.fn = SoftSignFunction.apply

    def forward(self, inputs:torch.Tensor): 
        return self.fn(inputs)