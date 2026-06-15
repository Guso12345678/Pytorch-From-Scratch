import torch
class L1LossFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, inputs:torch.Tensor,targets:torch.Tensor):
        ctx.save_for_backward(inputs,targets)
        outputs = torch.abs(inputs - targets)
        return outputs
    @staticmethod
    def backward(ctx, grad_outputs):
        inputs,targets = ctx.saved_tensors
        grad_inputs = torch.sign(inputs - targets)
        return grad_inputs * grad_outputs, None 
        