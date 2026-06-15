import torch 

class ReLU6Function(torch.autograd.Function): 
    
    @staticmethod
    def forward(ctx:any,inputs:torch.Tensor):
        mask1 = inputs >= 6
        mask2 = (inputs >= 0) & ( inputs < 6) 
        mask3 = inputs < 0 

        outputs = torch.empty_like(inputs)
        outputs[mask1] = 6  
        outputs[mask2] = inputs[mask2]
        outputs[mask3] = 0
        ctx.save_for_backward(mask1,mask2,mask3) 
        return outputs 
    
    @staticmethod
    def backward(ctx,grad_outputs):
        mask1, mask2, mask3 = ctx.saved_tensors 
        grad_inputs = torch.empty_like(grad_outputs)
        grad_inputs[mask1] = 0 
        grad_inputs[mask2] = grad_outputs[mask2]
        grad_inputs[mask3] = 0 
        return grad_inputs  



class ReLu6(torch.nn.Module): 
    def __init__(self): 
        super().__init__()
        self.fn = ReLU6Function.apply
    def forward(self,inputs) -> torch.Tensor: 
        return self.fn(inputs)