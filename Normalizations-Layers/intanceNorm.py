import torch
"""
    VOY A USAR LA NOTACION DE LOS TENSORES SIGUEN LA FORMA DE [BATCH, CHANNELS, HEIGHT, WIDTH]
"""
class InstanceNormFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx:any,inputs:torch.Tensor,eps:float,gamma:torch.Tensor,beta:torch.Tensor):
        outputs_pred = inputs.clone()
        mean:torch.Tensor = torch.mean(inputs,dim=(2,3),keep_dim=True)
        var:torch.Tensor = torch.var(inputs,dim=(2,3),keep_dim=True)
        outputs_pred = (outputs_pred - mean)/(torch.sqrt(var + eps))
        ctx.save_for_backward(inputs,outputs_pred,gamma,beta,var)
        ctx.eps = eps 
        outputs = gamma * outputs_pred - beta 
        return outputs
    @staticmethod
    def backward(ctx:any,grad_outputs:torch.Tensor): 
        _,output_pred,gamma,_,var = ctx.saved_tensors
        eps = ctx.eps
        ###PRIMERO: CALCULAMOS LOS GRADIENTES DE LOS PESOS ENTRENABLES
        grad_gamma = torch.sum(grad_outputs*output_pred, dim=(0,2,3)) ###CLAROOOOOO POR QUE SINO NO TE VA A HACER MATCHING DE LAS DIMENSIONES PORQUE LA SALIDA DEL GRADIENTE DE LA GAMMA TIENE QUE SER DE SHAPE 1,C,1,1
        grad_beta = torch.sum(grad_outputs, dim=(0,2,3)) ###CLAROOOOOO POR QUE SINO NO TE VA A HACER MATCHING DE LAS DIMENSIONES PORQUE LA SALIDA DEL GRADIENTE DE LA BETA TIENE QUE SER DE SHAPE 1,C,1,1
        ###SEGUNDO: CALCULAMOS LOS GRADIENTES DEL OUTPUT_PRED
        grad_output_pred = grad_outputs * gamma

        ###TERCERO: CALCULAMOS LOS GRADIENTES DE LOS INPUTS

        grad_inputs = (1 / torch.sqrt(var + eps)) * (grad_output_pred - torch.mean(grad_output_pred, dim=(2,3), keepdim=True) - output_pred * torch.mean(grad_output_pred * output_pred, dim=(2,3), keepdim=True))
        
        return grad_inputs, None, grad_gamma, grad_beta, None, None

class InstanceNorm(torch.nn.Module): 
    def __init__(self,num_channels:int,eps:float=1e-5): 
        self.eps = eps 
        self.gamma = torch.nn.Parameter(torch.empty(1,num_channels,1,1))
        self.beta = torch.nn.Parameter(torch.empty(1,num_channels,1,1))
        torch.nn.init.kaiming_uniform_(self.gamma)
        torch.nn.init.kaiming_uniform_(self.beta)
        self.fn = InstanceNormFunction.apply
    def forward(self,inputs:torch.Tensor): 
        return self.fn(inputs,self.eps,self.gamma,self.beta)