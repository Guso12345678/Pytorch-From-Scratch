import torch 
class ELUFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx:any,inputs:torch.Tensor,alpha:float): 
        ctx.save_for_backward(inputs)
        ctx.alpha = alpha 
        outputs = inputs.clone()
        outputs[inputs <= 0] = alpha*(torch.exp(inputs[inputs <= 0]) - 1)
        return outputs
    @staticmethod
    def backward(ctx:any,grad_outputs:torch.Tensor): 
        inputs, = ctx.saved_tensors
        alpha = ctx.alpha

        grad_inputs = grad_outputs.clone()
        grad_inputs[inputs <= 0] *= alpha*(torch.exp(inputs[inputs <= 0]))
        return grad_inputs, None

class ELU(torch.nn.Module): 
    def __init__(self,alpha:float=1.0): 
        super().__init__()
        self.fn = ELUFunction.apply
        self.alpha=alpha
    def forward(self,inputs:torch.Tensor): 
        return self.fn(inputs,self.alpha)
