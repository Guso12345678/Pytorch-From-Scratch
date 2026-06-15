import torch 
class BCEWithLogitsLossFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, inputs, targets, weight, eps): 
        ctx.save_for_backward(inputs,targets,weight)
        ctx.eps = eps

        #Definimos la sigmoide: 
        logits = (1) / (1 + torch.exp(-inputs))

        #Apicamos la formula con la sigmoide: 
        outputs = -weight * ((targets*torch.log(logits)) + ((1-targets)*torch.log(1-logits)))

        return outputs


        
    @staticmethod
    def backward(ctx, grad_output): 
        inputs, targets, weight = ctx.saved_tensors
        eps = ctx.eps

        logits = (1) / (1 + torch.exp(-inputs))
        #Sacamos el gradiente con respecto a los inputs
        grad_inputs = -weight * ((targets*(1 + torch.exp(-inputs)) - 1)/(1 + torch.exp(-inputs)))*grad_output

        #Sacamos el gradiente con respecto a los weights 

        grad_weight = -(targets*torch.log(logits) + (1 - targets)*torch.log(1 - logits))

        return grad_inputs, None, grad_weight, None




class BCEWithLogitsLoss(torch.nn.module): 
    def __init__(self,eps:float=1e12): 
        super().__init__()
        self.eps = eps

        self.weight = torch.nn.Parameter(torch.empty(2,))
        torch.nn.init.xavier_uniform_(self.weight)

        self.fn = BCEWithLogitsLossFunction.apply
        return None 
    def forward(self,inputs:torch.Tensor,targets:torch.Tensor): 
        return self.fn(inputs,targets,self.weight,self.eps)
    


###IMPLEMENTACION CHAT: 
# import torch
# from typing import Optional

# class BCEWithLogitsLossFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, inputs, targets, weight=None, pos_weight=None, eps=1e-12):
#         ctx.save_for_backward(inputs, targets, weight, pos_weight)
#         ctx.eps = eps
        
#         # Cálculo estable de la sigmoide
#         max_val = torch.clamp(-inputs, min=0)
#         log_sigmoid = inputs - (inputs * (1 - max_val.byte().float())) - \
#                      (max_val + torch.log(torch.exp(-max_val) + torch.exp(-inputs - max_val)))
        
#         # Cálculo de la pérdida
#         loss = (1 - targets) * inputs + log_sigmoid
#         if pos_weight is not None:
#             loss = loss * (pos_weight - 1) * targets + loss
#         if weight is not None:
#             loss = loss * weight
            
#         return -loss.mean()

#     @staticmethod
#     def backward(ctx, grad_output):
#         inputs, targets, weight, pos_weight = ctx.saved_tensors
#         eps = ctx.eps
        
#         # Cálculo estable del gradiente
#         sigmoid = torch.sigmoid(inputs)
#         grad_inputs = (sigmoid - targets)
#         if pos_weight is not None:
#             grad_inputs = grad_inputs * (pos_weight - 1) * targets + grad_inputs
#         if weight is not None:
#             grad_inputs = grad_inputs * weight
            
#         grad_inputs = grad_inputs * grad_output / inputs.size(0)
        
#         # Gradientes para weight y pos_weight (opcionales)
#         grad_weight = None
#         grad_pos_weight = None
        
#         return grad_inputs, None, grad_weight, grad_pos_weight, None

# class BCEWithLogitsLoss(torch.nn.Module):
#     def __init__(self, weight: Optional[torch.Tensor] = None, 
#                  pos_weight: Optional[torch.Tensor] = None, 
#                  reduction: str = 'mean', eps: float = 1e-12):
#         super().__init__()
#         self.register_buffer('weight', weight)
#         self.register_buffer('pos_weight', pos_weight)
#         self.reduction = reduction
#         self.eps = eps
        
#     def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         loss = BCEWithLogitsLossFunction.apply(
#             inputs, targets, self.weight, self.pos_weight, self.eps)
            
#         if self.reduction == 'sum':
#             return loss.sum()
#         elif self.reduction == 'none':
#             return loss
#         return loss.mean()