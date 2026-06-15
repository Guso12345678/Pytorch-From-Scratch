import torch 

class ReLU6Function(torch.autograd.Function): 
    
    @staticmethod
    def forward(ctx:any,inputs:torch.Tensor): 
        outputs = inputs.clone()
        ctx.save_for_backward(inputs)
        outputs[inputs >= 6] = 6 
        outputs[inputs <= 0] = 0 
        return outputs
    
    @staticmethod 
    def backward(ctx,grad_outputs): 
        inputs, = ctx.saved_tensors
        grad_inputs = grad_outputs.clone()
        grad_inputs[(inputs > 0)&(inputs < 6)] *= 1 
        grad_inputs[(inputs <= 0) | (inputs >=6)] *=0
        return grad_inputs
    


class ReLu6(torch.nn.Module): 
    def __init__(self): 
        super().__init__()
        self.fn = ReLU6Function.apply
    def forward(self,inputs) -> torch.Tensor: 
        return self.fn(inputs)
