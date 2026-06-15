import torch 
"""

    VOY A USAR LA NOTACION DE LOS TENSORES SIGUEN LA FORMA DE [BATCH, CHANNELS, HEIGHT, WIDTH]

"""
class BatchNormFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx:any,inputs:torch.Tensor,eps:float,gamma:torch.Tensor,beta:torch.Tensor):
        outputs_predict = inputs.clone()
        mean:torch.Tensor = torch.mean(inputs,dim=(0,2,3),keepdim=True)
        var:torch.Tensor = torch.var(inputs,dim=(0,2,3),keepdim=True)
        outputs_predict = (outputs_predict - mean)/(torch.sqrt(var + eps))
        ctx.save_for_backward(inputs,outputs_predict, gamma, beta, var)
        ctx.eps = eps
        outputs = (gamma * outputs_predict) + beta 
        return outputs
    
    @staticmethod
    def backward(ctx:any,grad_output:torch.Tensor): 
        _ ,output_predict, gamma, beta, var = ctx.saved_tensors
        eps = ctx.eps
        ###PRIMERO: CALCULAMOS LOS GRADIENTES DE LOS PESOS ENTRENABLES 
        grad_gamma = torch.sum(grad_output*output_predict, dim=(0,2,3))
        grad_beta = torch.sum(grad_output, dim=(0,2,3))

        ###SEGUNDO: CALCULAMOS LOS GRADIENTES DEL OUTPUT_PRED
        grad_output_pred = grad_output * gamma

        ###TERCERO: CALCULAMOS LOS GRADIENTES DE LOS INPUTS

        grad_inputs = (1 / torch.sqrt(var + eps)) * (grad_output_pred - torch.mean(grad_output_pred, dim=(0, 2, 3), keepdim=True) - output_predict * torch.mean(grad_output_pred * output_predict, dim=(0, 2, 3), keepdim=True))
        
        return grad_inputs, None, grad_gamma, grad_beta, None, None


class BatchNormalization(torch.nn.Module): 
    def __init__(self,num_channels,eps:float=1e-5): 
        super().__init__()
        self.eps = eps
        self.alpha = torch.nn.Parameter(torch.empty(1,num_channels))
        self.beta = torch.nn.Parameter(torch.empty(1,num_channels))
        torch.nn.init.kaiming_uniform_(self.alpha)
        torch.nn.init.kaiming_uniform_(self.beta)
        self.fn = BatchNormFunction.apply
    def forward(self,inputs:torch.Tensor): 
        return self.fn(inputs,self.eps,self.alpha,self.beta)

