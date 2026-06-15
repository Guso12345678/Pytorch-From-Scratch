import torch

class HardTanhFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs:torch.Tensor,max_value:float, min_value:float): 
        ctx.save_for_backward(inputs)
        ctx.max_value = max_value 
        ctx.min_value = min_value 

        mask1 = inputs > max_value
        mask2 = inputs < min_value
        mask3 = (inputs > min_value) & (inputs < max_value)

        outputs = torch.zeros_like(inputs)

        outputs[mask1] = max_value 
        outputs[mask2] = min_value
        outputs[mask3] = inputs[mask3]

        return outputs
    
    @staticmethod
    def backward(ctx,grad_outputs): 
        inputs, = ctx.saved_tensors
        max_value = ctx.max_value
        min_value = ctx.min_value 

        mask1 = inputs > max_value
        mask2 = inputs < min_value
        mask3 = (inputs > min_value) & (inputs < max_value) 

        grad_inputs = torch.zeros_like(grad_outputs)

        grad_inputs[mask1] = 0 
        grad_inputs[mask2] = 0 
        grad_inputs[mask3] = 1 

        return grad_inputs* grad_outputs, None, None 


