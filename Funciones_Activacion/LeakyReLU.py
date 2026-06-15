# import torch
# class LeakyReLUFunction(torch.autograd.Function): 
#     @staticmethod
#     def forward(ctx:any,inputs:torch.Tensor,alpha:float):
#         outputs = inputs.clone()
#         mascara_back = torch.ones(outputs.shape)
#         mascara_back[outputs <= 0] = alpha 
#         ctx.save_for_backward(mascara_back)
#         return outputs * mascara_back
#     def backward(ctx:any,grad_output:torch.Tensor): 
#         derivada_leaky, = ctx.saved_tensors
#         return grad_output * derivada_leaky

# class LeakyReLu(torch.nn.Module): 
#     def __init__(self,alpha=0.01): 
#         self.alpha:float = alpha
#         self.fn = LeakyReLUFunction.apply
#     def forward(self,inputs) -> torch.Tensor: 
#         return self.fn(inputs,self.alpha)



import torch

class LeakyReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, alpha: float):
        ctx.save_for_backward(inputs)
        ctx.alpha = alpha
        outputs = inputs.clone()
        outputs[inputs <= 0] *= alpha
        return outputs
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        alpha = ctx.alpha
        grad_inputs = grad_output.clone() 
        grad_inputs[inputs <= 0] *= alpha
        return grad_inputs, None

class LeakyReLU(torch.nn.Module):
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha 
        self.fn = LeakyReLUFunction.apply
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fn(inputs, self.alpha)