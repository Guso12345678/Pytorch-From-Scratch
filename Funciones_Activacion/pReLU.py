import torch 

class pReLUFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx:any, inputs:torch.Tensor, alpha:torch.Tensor): 
        ctx.save_for_backward(inputs,alpha)
        outputs = inputs.clone()
        outputs[inputs <= 0] *= alpha.item()
        
        return outputs
    @staticmethod
    def backward(ctx:any,grad_outputs:torch.Tensor): 
        inputs, alpha  = ctx.saved_tensors
        grad_inputs = grad_outputs.clone()
        grad_inputs[inputs <= 0]*= alpha.item()
        grad_alpha = torch.sum(grad_outputs*inputs[inputs<=0])
        return grad_inputs,grad_alpha

class pReLU(torch.nn.Module):  
    def __init__(self): 
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.empty(1))
        torch.nn.init.kaiming_uniform_(self.alpha)
        self.fn = pReLUFunction.apply
    
    def forward(self,inputs): 
        return self.fn(inputs,self.alpha)





    