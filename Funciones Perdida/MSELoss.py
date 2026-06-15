import torch 

class MSELossFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs:torch.Tensor,targets:torch.Tensor):
        ctx.save_for_backward(inputs,targets)
        loss = (inputs - targets)**2
        return loss 
    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = 2*(inputs - targets)
        return grad_inputs *grad_outputs, None