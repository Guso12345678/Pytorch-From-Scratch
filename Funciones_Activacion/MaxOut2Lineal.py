# deep learning libraries
import torch

# other libraries
import math
from typing import Any


class MaxoutFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the Maxout.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weights_first: torch.Tensor,
        bias_first: torch.Tensor,
        weights_second: torch.Tensor,
        bias_second: torch.Tensor,
    ) -> torch.Tensor:
        """
        This is the forward method of the relu.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [batch, input dim].

        Returns:
            outputs tensor. Dimensions: [batch, output dim].
        """
        #B,I_dim = inputs.shape
        outputs1 = inputs @ weights_first.T + bias_first 
        outputs2 = inputs @ weights_second.T + bias_second 

        mask1_outputs1_mas_igual_outputs2 = outputs1 >= outputs2
        mask2_outputs2_mas_outputs2 = outputs2 > outputs1

        outputs = outputs1*mask1_outputs1_mas_igual_outputs2 + outputs2*mask2_outputs2_mas_outputs2
        ctx.save_for_backward(inputs,weights_first,weights_second,mask1_outputs1_mas_igual_outputs2,mask2_outputs2_mas_outputs2)
        return outputs 


    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This method is the backward of the Maxout.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions: 
                [batch, output dim].

        Returns:
            inputs gradients. Dimensions: [batch, input dim].
            gradients for the first weights. Dimensions:
                [output dim, input dim].
            gradients for the first bias. Dimensions: [output dim].
            gradient for the second weights. Dimensions: [output dim,
                input dim].
            gradient for the second bias. Dimensions: [output dim].
        """
        inputs,weights1,weights2,mask1, mask2 = ctx.save_tensors 
        grad_output_1 = grad_output * mask1 #Te guardas unicamente los valores por los que pasa gradiente de la primera lineal.  
        grad_output_2 = grad_output * mask2 #Te guardas unicamente los valores por los que pasa gradiente de la segunda lineal. 

        #PRIMERO SACAMOS LO RELACIONADO CON LA PRIMERA LINEAL.  
        grad_weights1 = grad_output_1.T @ inputs #Obtenemos un shape de: [output dim, input dim]
        grad_bias1 = torch.sum(grad_output_1,dim=0) # Obtenemos un shape de: [output dim]
        grad_inputs1 = grad_output_1 @ weights1 # Obtenemos un shape de: [batch, input dim] 
        #SEGUNDO SACAMOS LO RELACIONADO CON LA SEGUNDA LINEAL: 
        grad_weights2 = grad_output_2.T @ inputs #Obtenemos un shape de: [output dim, input dim]
        grad_bias2 = torch.sum(grad_output_2, dim=0) #Obtenemos un shape de: [output dim]
        grad_inputs2 = grad_output_2 @ weights2 #Obtenemos un shape de [batch, input dim]

        #SACAMOS EL GRAD_INPUTS QUE SERA LA SUMA DE AMBOS GRADIENTES: 
        grad_inputs = grad_inputs1 + grad_inputs2 

        return grad_inputs, grad_weights1, grad_bias1, grad_weights2, grad_bias2



