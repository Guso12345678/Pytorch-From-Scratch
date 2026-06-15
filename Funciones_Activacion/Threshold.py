import torch 

class ThresholdFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs:torch.Tensor,threshold:float): 
        ctx.save_for_backward(inputs)
        ctx.threshold = threshold 
        mask1 = inputs > threshold
        mask2 = inputs <= threshold
        outputs = torch.zeros_like(inputs)
        outputs[mask1] = inputs[mask1]
        outputs[mask2] = threshold
        return outputs 
    
    @staticmethod
    def backward(ctx,grad_outputs): 
        inputs, = ctx.saved_tensors
        threshold = ctx.threshold 
        mask1 = inputs > threshold
        mask2 = inputs <= threshold 
        grad_inputs = torch.zeros_like(grad_outputs)
        grad_inputs[mask1] = 1 
        grad_inputs[mask2] = 0 
        return grad_inputs * grad_outputs, None  