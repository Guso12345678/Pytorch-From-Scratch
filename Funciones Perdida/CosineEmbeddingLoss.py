import torch 
from typing import Any

class CosineEmbeddingLossFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx: Any, inputs1:torch.Tensor, inputs2:torch.Tensor, target:torch.Tensor, margin:float): 
        ctx.save_for_backward(inputs1,inputs2,target)
        ctx.margin = margin

        #Calculo usando las formulas 
        output = inputs1.clone()
        mascara_1 = target == 1
        cos_x1_x2 = (inputs1 * inputs2)/(torch.sqrt(torch.sum(inputs1**2))) * (torch.sqrt(torch.sum(inputs2 ** 2)))
        mascara_2 = (target == -1) & (cos_x1_x2  - margin) > 0  
        mascara_3 = (target == -1) & (cos_x1_x2 - margin) < 0

        output[mascara_1] = 1 - (inputs1[mascara_1] * inputs2[mascara_1])/(torch.sqrt(torch.sum(inputs1[mascara_1]**2))) * (torch.sqrt(torch.sum(inputs2[mascara_1] ** 2)))

        output[mascara_2] = ((inputs1[mascara_2] * inputs2[mascara_2])/(torch.sqrt(torch.sum(inputs1[mascara_2]**2))) * (torch.sqrt(torch.sum(inputs2[mascara_2] ** 2))) - margin)
        
        output[mascara_3] = 0 

        return output
    
    @staticmethod 
    def backward(ctx:Any, grad_outputs:torch.Tensor): 
        inputs1, inputs2, target = ctx.saved_tensors 
        margin = ctx.margin 

        ##Clonamos los inputs:
        cos_x1_x2 =  (inputs1 * inputs2)/((torch.sqrt(torch.sum(inputs1**2))) * (torch.sqrt(torch.sum(inputs2 ** 2))))
        grad_inputs1 = grad_outputs.clone()
        grad_inputs2 = grad_outputs.clone()
        mascara_1 = target == 1
        mascara_2 = target == -1 & (cos_x1_x2  - margin) > 0
        mascara_3 = target == -1 & (cos_x1_x2 - margin) < 0

        ##Sacamos los inputs con respecto 
        #El gradiente de los inputs 1: 
        grad_inputs1[mascara_1] *= ((inputs2[mascara_1])/((torch.sqrt(torch.sum(inputs1[mascara_1] ** 2))) * (torch.sqrt(torch.sum(inputs2[mascara_1] ** 2))))) - (inputs1[mascara_1]*cos_x1_x2[mascara_1] / torch.sum(inputs1[mascara_1] ** 2))
        grad_inputs1[mascara_2] *= ((-inputs2[mascara_2])/((torch.sqrt(torch.sum(inputs1[mascara_2] ** 2))) * (torch.sqrt(torch.sum(inputs2[mascara_2] ** 2))))) + (inputs1[mascara_2]*cos_x1_x2[mascara_2] / torch.sum(inputs1[mascara_2] ** 2))
        grad_inputs1[mascara_3] *= 0 
        #El gradiente de los inputs 2: 
        grad_inputs2[mascara_1] *= ((inputs1[mascara_1])/((torch.sqrt(torch.sum(inputs1[mascara_1] ** 2))) * (torch.sqrt(torch.sum(inputs2[mascara_1] ** 2))))) - (inputs2[mascara_1]*cos_x1_x2[mascara_1] / torch.sum(inputs2[mascara_1] ** 2))
        grad_inputs2[mascara_2] *= ((-inputs1[mascara_2])/((torch.sqrt(torch.sum(inputs1[mascara_2] ** 2))) * (torch.sqrt(torch.sum(inputs2[mascara_2] ** 2))))) - (inputs2[mascara_2]*cos_x1_x2[mascara_2] / torch.sum(inputs2[mascara_2] ** 2))
        grad_inputs2[mascara_3] *= 0 

        return grad_inputs1, grad_inputs2, None, None
class CosineEmbeddingLoss(torch.nn.module):
     
    def __init__(self,margin:float=0.5): 
        super().__init__()
        self.margin = margin
        self.fn = CosineEmbeddingLossFunction.apply
        return None 
    
    def forward(self,inputs1:torch.Tensor,inputs2:torch.Tensor,targets:torch.Tensor): 
        return self.fn(inputs1, inputs2,targets,self.margin)
    




##IMPLEMETACION DE CHAT DEL FORWARD: 
# @staticmethod
# def forward(ctx: Any, inputs1: torch.Tensor, inputs2: torch.Tensor, 
#             target: torch.Tensor, margin: float): 
#     ctx.save_for_backward(inputs1, inputs2, target)
#     ctx.margin = margin

#     # Calcular producto punto y normas manualmente
#     dot_product = (inputs1 * inputs2).sum(dim=1, keepdim=True)
#     norm1 = torch.sqrt((inputs1**2).sum(dim=1, keepdim=True) + 1e-8)
#     norm2 = torch.sqrt((inputs2**2).sum(dim=1, keepdim=True) + 1e-8)
    
#     # Calcular similitud coseno
#     cos_sim = dot_product / (norm1 * norm2)
    
#     # Inicializar output
#     output = torch.zeros_like(cos_sim)
    
#     # Máscaras para diferentes casos
#     mask_pos = (target == 1).view_as(output)  # Pares similares
#     mask_neg = (target == -1).view_as(output) # Pares disímiles
    
#     # Caso y = 1: 1 - cos(x1, x2)
#     output[mask_pos] = 1 - cos_sim[mask_pos]
    
#     # Caso y = -1: max(0, cos(x1, x2) - margin)
#     # Implementamos max(0, x) manualmente
#     diff = cos_sim - margin
#     positive_diff = diff * (diff > 0).float()  # Equivalente a max(0, diff)
#     output[mask_neg] = positive_diff[mask_neg]
    
#     return output