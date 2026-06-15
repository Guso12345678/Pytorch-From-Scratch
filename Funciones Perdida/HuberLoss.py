import torch

class HuberLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs:torch.Tensor, targets:torch.Tensor, delta):
        ctx.save_for_backward(inputs,targets)
        ctx.delta = delta 
        mask1 = torch.abs(inputs-targets) < delta 
        mask2 = torch.abs(inputs-targets) >= delta 
        outputs = torch.zeros_like(inputs)
        outputs[mask1] = 0.5*(inputs[mask1] - targets[mask1])**2
        outputs[mask2] = delta*(torch.abs(inputs[mask2] - targets[mask2]) - 0.5*delta)
        return outputs 
    
    @staticmethod
    def backward(ctx,grad_output): 
        inputs, targets = ctx.saved_tensors 
        delta = ctx.delta 
        grad_inputs = torch.zeros_like(inputs)
        mask1 = torch.abs(inputs-targets) < delta
        mask2 = torch.abs(inputs-targets) >= delta
        grad_inputs[mask1] = inputs[mask1] - targets[mask1]
        grad_inputs[mask2] = delta*torch.sign(inputs[mask2] - targets[mask2])
        return grad_inputs * grad_output, None, None 