import torch 
"""[batch, channels, height, width]."""
class GroupNormFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx:any,inputs:torch.Tensor,gamma:torch.Tensor,beta:torch.Tensor,num_grupos:int,eps:float): 
        outputs = inputs.clone()
        N, C, H, W = outputs.shape 
        #PRIMER PASO VIEW DEL OUTPUTS PARA QUE MACHEEN LAS COSAS: 
        outputs_shapeados = outputs.view(N,num_grupos,C//num_grupos,H,W)
        mean:torch.Tensor = torch.mean(outputs_shapeados,dim=(2,3,4),keepdim=True)
        var:torch.Tensor = torch.var(outputs_shapeados,dim=(2,3,4),keepdim=True)
        outputs_shapeados_normalizados = (outputs_shapeados - mean)/(torch.sqrt(var + eps))
        ctx.save_for_backward(inputs,outputs_shapeados_normalizados,gamma,beta,var)
        ctx.eps = eps 
        ctx.num_grupos = num_grupos
        ctx.num_canales = C
        outputs_finales = (gamma * outputs_shapeados_normalizados) + beta 
        return outputs_finales.view(N,C,H,W)
    @staticmethod
    def backward(ctx:any,grad_outputs:torch.Tensor): 
        _,outputs_shapeados_nomralizados,gamma,_,var = ctx.saved_tensors
        eps = ctx.eps 
        num_grupos = ctx.num_grupos
        num_canales = ctx.num_canales
        grad_outputs_shapeados = grad_outputs.view(grad_outputs.size(0),num_grupos,num_canales//num_grupos,grad_outputs.size(2),grad_outputs.size(3))
        ###PRIMERO GRADIENTES DE LAS GAMMAS Y LAS BETAS QUE SERAN EN FUNCION DE LOS GRUPOS: 
        grad_gamma = torch.sum(grad_outputs_shapeados*outputs_shapeados_nomralizados,dim=(0,2,3,4),keepdim=True)
        grad_beta = torch.sum(grad_outputs_shapeados,dim=(0,2,3,4),keepdim=True)
        ###SEGUNDO CALCULAMOS LOS GRADIENTES PARA EL OUTPUT NORMALIZADO 
        grad_output_normalizado = grad_outputs_shapeados * gamma
        ###CALCULAMOS EL GRAD_INPUT
        grad_input = (1/torch.sqrt(var + eps))*(grad_output_normalizado - torch.mean(grad_output_normalizado,dim=(2,3,4),keepdim=True)- outputs_shapeados_nomralizados*torch.mean(grad_output_normalizado*outputs_shapeados_nomralizados,dim=(2,3,4),keepdim=True))
        
        return grad_input.view(grad_input.size(0),num_canales,grad_input.size(3),grad_input.size(4)),None,grad_gamma,grad_beta, None, None

class GroupNorm(torch.nn.Module): 
    def __init__(self,num_canales:int,num_grupos:int,eps:float=1e-5): 
        self.eps = eps
        self.num_canales = num_canales
        self.num_grupos = num_grupos
        self.gamma = torch.nn.Parameter(torch.empty(1,num_canales,1,1))
        self.beta = torch.nn.Parameter(torch.empty(1,num_canales,1,1))
        torch.nn.init.kaiming_uniform_(self.gamma)
        torch.nn.init.kaiming_uniform_(self.beta)
        self.fn = GroupNormFunction.apply 
    def forward(self,inputs:torch.Tensor): 
        return self.fn(inputs,self.gamma,self.beta,self.num_grupos,self.eps)
