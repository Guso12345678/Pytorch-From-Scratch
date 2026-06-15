import torch 

class SoftshrinkFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs:torch.Tensor,lambd:torch.Tensor):
        ctx.save_for_backward(inputs)
        ctx.lambd = lambd 
        mask1 = inputs > lambd 
        mask2 = inputs < -lambd 
        mask3 = (inputs > -lambd) & (inputs < lambd)
        outputs = torch.zeros_like(inputs)
        outputs[mask1] = inputs[mask1] - lambd
        outputs[mask2] = inputs[mask2] + lambd
        outputs[mask3] = 0 
        return outputs 
    
    @staticmethod
    def backward(ctx,grad_outputs): 
        inputs,  = ctx.saved_tensors
        lambd = ctx.lambd 
        mask = (inputs > lambd) | (inputs < -lambd)
        grad_inputs = torch.zeros_like(grad_outputs)
        grad_inputs[mask] = 1 
        return grad_inputs * grad_outputs, None 

