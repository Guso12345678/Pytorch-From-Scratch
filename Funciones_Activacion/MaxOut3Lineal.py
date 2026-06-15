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
        weights_third: torch.Tensor, 
        bias_third: torch.Tensor
    ) -> torch.Tensor:
        """
        This is the forward method of the relu.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [batch, input dim].

        Returns:
            outputs tensor. Dimensions: [batch, output dim].
        """
        outputs1 = inputs @ weights_first.T + bias_first 
        outputs2 = inputs @ weights_second.T + bias_second
        outputs3 = inputs @ weights_third.T + bias_third

        mask1 = (outputs1 >= outputs2) & (outputs1 >= outputs3)
        mask2 = (outputs2 > outputs1) & (outputs2 > outputs3)
        mask3 = (outputs3 > outputs1) & (outputs3 > outputs2)

        outputs = outputs1*mask1 + outputs2*mask2 + outputs3*mask3 
        ctx.save_for_backward(inputs,weights_first,weights_second,weights_third,mask1,mask2,mask3)

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
        inputs,weights1,weights2,weights3,mask1, mask2, mask3 = ctx.saved_tensors
        
        grad_output1 = grad_output * mask1
        grad_output2 = grad_output * mask2
        grad_output3 = grad_output * mask3 

        ###PRIMERO PARA LA PRIMERA LINEAL: 
        grad_weights1 = grad_output1.T @ inputs
        grad_bias1 = torch.sum(grad_output1,dim=0)
        grad_inputs1 = grad_output1 @ weights1

        ###SEGUNDO PARA LA SEGUNDA LINEAL: 
        grad_weights2 = grad_output2.T @ inputs 
        grad_bias2 = torch.sum(grad_output2,dim=0)
        grad_inputs2 = grad_output2 @ weights2

        ###TERCERO PARA LA TERCERA LINEAL: 
        grad_weights3 = grad_output3.T @ inputs 
        grad_bias3 = torch.sum(grad_output3,dim=0)
        grad_inputs3 = grad_output3 @ weights3

        ###SACAMOS EL GRADIENTE DE LOS INPUTS COMO LA SUMA DE LOS TRES: 
        grad_inputs = grad_inputs1 + grad_inputs2 + grad_inputs3

        return grad_inputs, grad_weights1, grad_bias1, grad_weights2, grad_bias2, grad_weights3, grad_bias3
    

