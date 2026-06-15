import torch 

class pReLUFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx:any, inputs:torch.Tensor, alpha:torch.Tensor):
        mask1 = inputs >= 0 
        mask2 = inputs < 0 
        outputs = torch.empty_like(inputs)
        outputs[mask1] = inputs[mask1]
        outputs[mask2] = alpha*inputs[mask2]
        ctx.save_for_backward(inputs,alpha,mask1,mask2)
        return outputs  
    @staticmethod
    def backward(ctx:any,grad_outputs:torch.Tensor):
        inputs, alpha, mask1, mask2 = ctx.saved_tensors 
        grad_inputs = grad_outputs.clone()
        grad_inputs[mask1] *= 1 
        grad_inputs[mask2] *= alpha 
        grad_alpha = torch.sum(grad_outputs[mask2]*inputs[mask2]) #Cuidado que no se me olvide multiplicar por el grad_outputs
        return grad_inputs, grad_alpha 
class pReLU(torch.nn.Module):  
    def __init__(self): 
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.empty(1))
        torch.nn.init.kaiming_uniform_(self.alpha)
        self.fn = pReLUFunction.apply
    
    def forward(self,inputs): 
        return self.fn(inputs,self.alpha)
