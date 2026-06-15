import torch

class HardSwishFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx:any, inputs:torch.Tensor):
        ctx.save_for_backward(inputs) 
        outputs = inputs.clone()
        outputs[inputs <= -3] = 0
        outputs[inputs >= 3] = inputs[inputs >= 3]
        outputs[(inputs > -3) & (inputs < 3)] = inputs[(inputs > -3) & (inputs < 3)] * (inputs[(inputs > -3) & (inputs < 3)] + 3) / 6
        return outputs

    @staticmethod
    def backward(ctx:any, grad_outputs:torch.Tensor): 
        inputs, = ctx.saved_tensors
        grad_inputs = grad_outputs.clone()
        grad_inputs[inputs <= -3] = 0
        grad_inputs[inputs >= 3] = grad_outputs[inputs >= 3]
        grad_inputs[(inputs > -3) & (inputs < 3)] *= (inputs[(inputs > -3) & (inputs < 3)] / 3 + 0.5)

        return grad_inputs, None

class HardSwish(torch.nn.Module): 
    def __init__(self): 
        super().__init__()
        self.fn = HardSwishFunction.apply

    def forward(self, inputs:torch.Tensor): 
        return self.fn(inputs)

        