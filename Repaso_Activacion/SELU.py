import torch 

class SELUFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx:any,inputs:torch.Tensor,alpha:float,lamda:float): 
        mask1 = inputs > 0 
        mask2 = inputs <= 0 
        outputs = torch.empty_like(inputs)
        outputs[mask1] = lamda*inputs[mask1]
        outputs[mask2] = lamda*alpha*(torch.exp(inputs[mask2])-1)
        ctx.alpha = alpha 
        ctx.lamda = lamda 
        ctx.save_for_backward(inputs,mask1,mask2)
        return outputs  
    @staticmethod
    def backward(ctx:any,grad_outputs:torch.Tensor): 
        inputs, mask1, mask2= ctx.saved_tensors
        alpha = ctx.alpha 
        lamda = ctx.lamda 
        grad_inputs = torch.empty_like(grad_outputs)
        grad_inputs[mask1] = lamda*grad_outputs[mask1]
        grad_inputs[mask2] = lamda*alpha*torch.exp(inputs[mask2])*grad_outputs[mask2]
        return grad_inputs,None,None




class SELU(torch.nn.Module): 
    def __init__(self,alpha:float=1.6733,lamda:float=1.0507): 
        super().__init__()
        self.fn = SELUFunction.apply
        self.alpha = alpha
        self.lamda = lamda
    
    def forward(self,inputs:torch.Tensor): 
        return self.fn(inputs,self.alpha,self.lamda)