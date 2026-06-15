import torch 
"""

    VOY A USAR LA NOTACION DE LOS TENSORES SIGUEN LA FORMA DE [BATCH, CHANNELS, HEIGHT, WIDTH]

"""
class LayerNormFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx:any, inputs:torch.Tensor, eps:float, gamma:torch.Tensor, beta:torch.Tensor): 
        outputs_pred = inputs.clone()
        mean:torch.Tensor = torch.mean(inputs, dim=(1,2,3), keepdim=True)
        var:torch.Tensor = torch.var(inputs, dim=(1,2,3), keepdim=True)
        outputs_pred = (outputs_pred - mean) / (torch.sqrt(var + eps))
        ctx.save_for_backward(inputs , outputs_pred , gamma, beta, var)
        ctx.eps = eps
        outputs = outputs_pred * gamma + beta 
        return outputs

    @staticmethod
    def backward(ctx: any, grad_outputs: torch.Tensor):
        inputs, outputs_pred, gamma, var = ctx.saved_tensors
        eps = ctx.eps
        grad_gamma = torch.sum(grad_outputs * outputs_pred, dim=(0, 2, 3), keepdim=True)  
        grad_beta = torch.sum(grad_outputs, dim=(0, 2, 3), keepdim=True)                  
        grad_outputs_pred = grad_outputs * gamma
        m = inputs.size(1) * inputs.size(2) * inputs.size(3) 
        grad_inputs = (1 / torch.sqrt(var + eps)) * (grad_outputs_pred - torch.mean(grad_outputs_pred, dim=(1, 2, 3), keepdim=True) -outputs_pred * torch.mean(grad_outputs_pred * outputs_pred, dim=(1, 2, 3), keepdim=True))

        return grad_inputs, None, grad_gamma, grad_beta

        

class LayerNorm(torch.nn.Module): 
    def __init__(self,num_carac:int,eps:float=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.empty(1,num_carac))
        self.beta = torch.nn.Parameter(torch.empty(1,num_carac))
        torch.nn.init.kaiming_uniform_(self.gamma)
        torch.nn.init.kaiming_uniform_(self.beta)
        self.fn = LayerNormFunction.apply

    def forward(self,inputs:torch.Tensor): 
        return self.fn(inputs,self.eps,self.gamma,self.beta) 
