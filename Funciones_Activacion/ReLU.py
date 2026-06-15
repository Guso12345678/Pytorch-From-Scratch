import torch 

class ReLUFunction(torch.autograd.Function): 
    
    @staticmethod
    def forward(ctx:any,inputs:torch.Tensor): 
        outputs = inputs.clone()
        mascara_back = torch.zeros(inputs.shape)
        mascara_back[outputs > 0] = 1
        ctx.save_for_backward(mascara_back)
        return outputs * mascara_back
    
    @staticmethod 
    def backward(ctx,grad_outputs): 
        derivada_relu,  = ctx.saved_tensor
        return grad_outputs * derivada_relu
    


class ReLu(torch.nn.Module): 
    def __init__(self): 
        super().__init__()
        self.fn = ReLUFunction.apply
    def forward(self,inputs) -> torch.Tensor: 
        return self.fn(inputs)