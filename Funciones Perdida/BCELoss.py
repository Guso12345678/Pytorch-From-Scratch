from typing import Any 
import torch 

class BCELossFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx:Any,weight:torch.Tensor,inputs:torch.Tensor,targets:torch.Tensor,eps:float=1e12): 
        ctx.save_for_backward(inputs,targets,weight)
        ctx.eps = eps

        inputs = torch.clamp(inputs,eps,1-eps)

        output = -weight * ((targets * torch.log(inputs)) + ((1 - targets) * torch.log(1 - inputs)))

        return output

    @staticmethod
    def backward(ctx:Any,grad_output:torch.Tensor): 
        inputs, target, weight = ctx.saved_tensors
        eps = ctx.eps
        #Forzamos que no sea ni 0 ni 1: 
        inputs = torch.clamp(inputs, eps, 1-eps)
        #Sacamos el gradiente de los inputs,  
        grad_inputs = weight * ((inputs - target) / (inputs*(1 - inputs)))
        grad_inputs = grad_inputs * grad_output  

        #Sacamos el gradiente de los pesos (opcional)
        grad_weight = (-target*torch.log(inputs)) - ((1 - target)*torch.log(1 - inputs))
        grad_weight = grad_weight.mean(dim=0) * grad_output  
        return grad_inputs, None ,grad_weight, None

class BCELoss(torch.nn.module): 
    def __init__(self): 
        super().__init__()
        self.fn = BCELossFunction.apply
        return None 
    def forward(self,weight: torch.Tensor,inputs:torch.Tensor,targets:torch.Tensor,eps:float): 
        return self.fn(weight,inputs,targets,eps)
    


##IMPLEMENTACION DE CHAT: 
# import torch
# from typing import Any

# class BCELossFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx: Any, inputs: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor = None, eps: float = 1e-12) -> torch.Tensor:
#         # Guardar tensores para backward
#         ctx.save_for_backward(inputs, targets, weight)
#         ctx.eps = eps
        
#         # Clamp para estabilidad numérica
#         inputs_clamped = torch.clamp(inputs, eps, 1 - eps)
        
#         # Calcular pérdida
#         loss = - (targets * torch.log(inputs_clamped) + (1 - targets) * torch.log(1 - inputs_clamped))
        
#         # Aplicar pesos si existen
#         if weight is not None:
#             loss = loss * weight
            
#         return loss.mean()  # Reducción por defecto: media

#     @staticmethod
#     def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
#         inputs, targets, weight = ctx.saved_tensors
#         eps = ctx.eps
        
#         # Clamp para estabilidad numérica
#         inputs_clamped = torch.clamp(inputs, eps, 1 - eps)
#         batch_size = inputs.size(0)
        
#         # Gradiente respecto a inputs
#         grad_inputs = (inputs_clamped - targets) / (inputs_clamped * (1 - inputs_clamped))
        
#         # Aplicar pesos si existen
#         if weight is not None:
#             grad_inputs = grad_inputs * weight
        
#         # Normalizar por tamaño del batch y multiplicar por grad_output
#         grad_inputs = grad_inputs * (grad_output / batch_size)
        
#         # Gradiente respecto a weight (solo si weight fue proporcionado)
#         grad_weight = None
#         if weight is not None:
#             grad_weight = (-targets * torch.log(inputs_clamped) - ((1 - targets) * torch.log(1 - inputs_clamped))
#             grad_weight = grad_weight.mean(dim=0) * grad_output  # Promedio sobre el batch
            
#         return grad_inputs, None, grad_weight, None  # None para eps

# class BCELoss(torch.nn.Module):
#     def __init__(self, weight: torch.Tensor = None, reduction: str = 'mean'):
#         super().__init__()
#         self.weight = weight
#         self.reduction = reduction
#         self.validate_reduction()
        
#     def validate_reduction(self):
#         if self.reduction not in ['mean', 'sum', 'none']:
#             raise ValueError(f"Reduction must be 'mean', 'sum' or 'none', got {self.reduction}")
            
#     def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         # Aplicar la función de pérdida
#         loss = BCELossFunction.apply(inputs, targets, self.weight)
        
#         # Manejar reducción
#         if self.reduction == 'sum':
#             return loss.sum()
#         elif self.reduction == 'none':
#             return loss.view_as(inputs)
#         return loss  