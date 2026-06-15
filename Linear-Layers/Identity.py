import torch 

class IdentityFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs): 
        """
            The Identity function does nothing only pass the inputs:
            The Outputs has te same shape. 
        """
        mask_for_backward = torch.ones_like(inputs) #Not necessary
        ctx.save_for_backward(mask_for_backward)
        return inputs 
    
    @staticmethod
    def backward(ctx,grad_outputs): 
        """
            the grad_inputs same shape as grad_outputs  
        """
        mask, = ctx.saved_tensors
        grad_inputs = grad_outputs * mask 
        return grad_inputs 