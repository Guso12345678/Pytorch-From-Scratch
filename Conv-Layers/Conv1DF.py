import torch
from typing import Any
class Conv1DFunctional(torch.autograd.Function): 
    @staticmethod
    def forward(ctx:Any,inputs:torch.Tensor,kernel:torch.Tensor): 
        #Primer Paso: Cambiar las dimensiones del tensor: 
        inputs_unfoldeados = torch.nn.functional.unfold(inputs.unsqueeze(2),kernel_size=(1,kernel.size(2)),dilation=(1,1),padding=0,stride=1)
        #Modificamos tambien los pesos para que se pueda llevar a cabo al multiplicacion: 
        kernel_shapeado = kernel.view(kernel.size(0),-1)
        ctx.save_for_backward(inputs_unfoldeados,kernel_shapeado)
        #Por ultimo hacemos la multiplicacion donde ya obtenemos las dimensiones que queremos: 
        output = kernel_shapeado @ inputs_unfoldeados
        return output
    
    @staticmethod
    def backward(ctx:Any,grad_output:torch.Tensor): 
        inputs_unfoldeados, kernel_shapeado = ctx.saved_tensors
        ###CALCULO DEL GRADIENTE DEL KERNEL
        grad_outputs_shapeados = grad_output.transpose(1,2)
        grad_kernel:torch.Tensor = inputs_unfoldeados @ grad_outputs_shapeados

        ###CALCULO DEL GRADIENTE DE LOS INPUTS
        grad_input_dim_mas = torch.nn.functional.fold(torch.matmul(grad_output,kernel_shapeado),(1,5),(1,3),dilation=(1,1),padding=0,stride=1)
        grad_input = grad_input_dim_mas.squeeze(2)
        return grad_input, grad_kernel.transpose(1,2) 
