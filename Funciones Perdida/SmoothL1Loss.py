import torch 
class SmoothL1LossFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs:torch.Tensor,targets:torch.Tensor,beta:float): 
        ctx.save_for_backward(inputs,targets)
        ctx.beta = beta 
        loss = torch.zeros_like(inputs)
        mask1 = torch.abs(inputs - targets) < beta 
        mask2 = torch.abs(inputs - targets) >= beta
        loss[mask1] = (0.5*(inputs[mask1] - targets[mask1])**2)/beta 
        loss[mask2] = torch.abs(inputs[mask2] -targets[mask2]) - 0.5*beta 
        return loss 
    @staticmethod
    def backward(ctx,grad_outputs:torch.Tensor): 
        inputs, targets = ctx.saved_tensors 
        beta = ctx.beta 
        mask1 = torch.abs(inputs - targets) < beta 
        mask2 = torch.abs(inputs - targets) >= beta 
        grad_inputs = torch.zeros_like(inputs)
        grad_inputs[mask1] = grad_outputs[mask1] * (inputs[mask1] - targets[mask1]) / beta
        grad_inputs[mask2] = grad_outputs[mask2] * torch.sign(inputs[mask2] - targets[mask2])


        return grad_inputs, None, None