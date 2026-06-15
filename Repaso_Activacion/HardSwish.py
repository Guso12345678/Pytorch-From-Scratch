import torch
class HardSwishFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx:any, inputs:torch.Tensor): 
        mask1 = inputs <= -3 
        mask2 = inputs >= 3 
        mask3 = (inputs > -3) & (inputs < 3)
        outputs = torch.zeros_like(inputs)
        outputs[mask1] = 0 
        outputs[mask2] = inputs[mask2]
        outputs[mask3] = inputs[mask3]*(inputs[mask3] + 3)/6
        ctx.save_for_backward(inputs,mask1,mask2,mask3) 
        return outputs 
    
    @staticmethod
    def backward(ctx,grad_outputs): 
        inputs, mask1, mask2, mask3 = ctx.saved_tensors 
        grad_inputs = torch.empty_like(grad_outputs)
        grad_inputs[mask1] = 0 
        grad_inputs[mask2] = grad_outputs[mask2]
        grad_inputs[mask3] = ((2 * inputs[mask3] + 3) / 6) * grad_outputs[mask3]
        return grad_inputs  
