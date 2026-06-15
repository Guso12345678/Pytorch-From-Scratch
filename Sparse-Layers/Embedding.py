# deep learning libraries
import torch

# other libraries
from typing import Any, Optional


class EmbeddingFuncion(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the Embedding layer.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        padding_idx: int,
    ) -> torch.Tensor:
        """
        This is the forward method of the Embedding layer.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [batch].

        Returns:
            outputs tensor. Dimensions: [batch, output dim].
        """
        #the weights are the shape: [tamaño_vocab,embedding_dim]
        outputs = weight[inputs,:] #Para que se quede unicamente con aquellos valores que le tocan. 
        mascara_zeros = torch.zeros_like(weight,dtype=weight.dtype)
        ctx.save_for_backward(inputs,weight,mascara_zeros,outputs,torch.tensor(padding_idx))
        return outputs 
    
    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[None, torch.Tensor, None]:
        """
        This method is the backward of the Embedding layer.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions:
                [batch, output dim].

        Returns:
            None value.
            inputs gradients. Dimensions: [batch].
            None value.
        """

        # TODO 
        inputs, weights,mascara_zeros ,outputs,padding_idx = ctx.saved_tensors 
        padding_idx = padding_idx.item()
        ###GRADIENTE DE LOS PESOS:  
        gradiente_pesos = grad_outputs.clone()
        ###SOLO FUNCIONA PARA SIEMPRE PERO MAS LIO: 

        inputs_unique_values, counts = torch.unique(inputs,return_counts=True)
        for i in range(len(inputs_unique_values)): 
            if inputs_unique_values[i] == padding_idx and padding_idx == None: 
                continue 
            else: 
                mascara_zeros[padding_idx] = 0 
            mascara_zeros[inputs_unique_values[i],:] = counts[i]*grad_outputs[inputs_unique_values[i],:]

        ###FUNCIONA PARA SIEMPRE: 
        # for i in range(inputs.shape[0]): 
        #     idx = inputs[i].item() 
        #     if idx == padding_idx : 
        #         continue
        #     mascara_zeros[idx] += grad_outputs[i]

        ##GRADIENTE DE LAS ENTRADAS 
        # grad_inputs = grad_outputs.clone()
        # grad_inputs[:,padding_idx] *= 0 #Queremos que unicamente pase el gradiente por aquellos
        # grad_inputs_final = grad_inputs * outputs 
        return None,mascara_zeros, None 




class Embedding(torch.nn.Module):
    """
    This is the class that represents the Embedding Layer.
    """

    padding_idx: int

    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ) -> None:
        """
        This method is the constructor of the Embedding layer.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim)
        )

        # init parameters corectly
        self.reset_parameters()

        # set padding idx
        self.padding_idx = padding_idx if padding_idx is not None else -1

        self.fn = EmbeddingFuncion.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [batch].

        Returns:
            outputs tensor. Dimensions: [batch, output dim].
        """

        return self.fn(inputs, self.weight, self.padding_idx)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight)