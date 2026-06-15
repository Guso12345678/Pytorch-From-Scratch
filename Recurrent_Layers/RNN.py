# deep learning libraries
import torch

# other libraries
import math
from typing import Any


class RNNFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the RNN.
    """

    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        inputs: torch.Tensor,
        h0: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: torch.Tensor,
        bias_hh: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This is the forward method of the RNN.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [batch, sequence,
                input size].
            h0: first hidden state. Dimensions: [1, batch,
                hidden size].
            weight_ih: weight for the inputs.
                Dimensions: [hidden size, input size].
            weight_hh: weight for the inputs.
                Dimensions: [hidden size, hidden size].
            bias_ih: bias for the inputs.
                Dimensions: [hidden size].
            bias_hh: bias for the inputs.
                Dimensions: [hidden size].


        Returns:
            outputs tensor. Dimensions: [batch, sequence,
                hidden size].
            final hidden state for each element in the batch.
                Dimensions: [1, batch, hidden size].
        """ 
        N,L,Hin = inputs.shape 
        Hout = h0.shape[-1]
        h_acumulacion = torch.empty(size=(N,L,Hout),device=h0.device,dtype=h0.dtype)
        mask_t = torch.empty(size=(N,L,Hout))

        for i in range(L): 
            h_prev = h0.squeeze(0) if i == 0 else h_acumulacion[:,i-1,:]
            gi = inputs[:,i,:] @ weight_ih.T + bias_ih
            gh = h_prev @ weight_hh.T + bias_hh 
            g = gi + gh 
            mask = g > 0 
            gt = g * mask 
            mask_t[:,i,:] = mask 
            h_acumulacion[:,i,:] = gt 
        ctx.save_for_backward(inputs,h0,weight_hh,weight_ih,bias_ih,bias_hh,h_acumulacion,mask_t)
        return h_acumulacion, h_acumulacion[:,-1,:].unsqueeze(0)
    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor, grad_hn: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        This method is the backward of the RNN.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions: [*].

        Returns:
            inputs gradients. Dimensions: [batch, sequence,
                input size].
            h0 gradients state. Dimensions: [1, batch,
                hidden size].
            weight_ih gradient. Dimensions: [hidden size,
                input size].
            weight_hh gradients. Dimensions: [hidden size,
                hidden size].
            bias_ih gradients. Dimensions: [hidden size].
            bias_hh gradients. Dimensions: [hidden size].
        """
        inputs, h0 ,weight_hh, weight_ih,bias_ih,bias_hh,h_acumulacion,mask_t = ctx.saved_tensors 
        N,L,Hin = inputs.shape 
        Hout = h0.shape[-1] 
        grad_inputs = torch.empty(size=(N,L,Hin),device=inputs.device,dtype=inputs.dtype)
        grad_h = torch.empty(size=(N,L + 1,Hout),device=h0.device,dtype=h0.dtype)
        grad_weight_ih = torch.empty(size=(Hout,L,Hin),device=weight_ih.device,dtype=weight_ih.dtype)
        grad_weight_hh = torch.empty(size=(Hout,L,Hout),device=weight_hh.device,dtype=weight_hh.dtype)
        grad_bias_ih = torch.empty(size=(L,Hout),device=bias_ih.device,dtype=bias_ih.dtype)
        grad_bias_hh = torch.empty(size=(L,Hout),device=bias_hh.device,dtype=bias_hh.dtype)
        grad_h[:,-1,:] = grad_hn.squeeze(0) 

        for i in reversed(range(L)): 
            h_prev = h0.squeeze(0) if i == 0 else h_acumulacion[:,i-1,:]
            grad_final =  (grad_h[:,i + 1,:] + grad_output[:,i,:])*mask_t[:,i,:] #[N,Hout]

            #Computamos primero el gradiente de los 
            grad_weight_ih[:,i,:] = grad_final.T @ inputs[:,i,:]
            grad_bias_ih[i,:] = torch.sum(grad_final,dim=0)
            grad_weight_hh[:,i,:] = grad_final.T @ h_prev 
            grad_bias_hh[i,:] = torch.sum(grad_final,dim=0)
            grad_h[:,i,:] = grad_final @ weight_hh 
            grad_inputs[:,i,:] = grad_final @ weight_ih 
        return grad_inputs, grad_h[:,0,:].unsqueeze(0), torch.sum(grad_weight_ih,dim=1),torch.sum(grad_weight_hh,dim=1) ,torch.sum(grad_bias_ih,dim=0), torch.sum(grad_bias_hh,dim=0)   

    






class RNN(torch.nn.Module):
    """
    This is the class that represents the RNN Layer.
    """

    def __init__(self, input_dim: int, hidden_size: int):
        """
        This method is the constructor of the RNN layer.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.hidden_size = hidden_size
        self.weight_ih: torch.Tensor = torch.nn.Parameter(
            torch.empty(hidden_size, input_dim)
        )
        self.weight_hh: torch.Tensor = torch.nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.bias_ih: torch.Tensor = torch.nn.Parameter(torch.empty(hidden_size))
        self.bias_hh: torch.Tensor = torch.nn.Parameter(torch.empty(hidden_size))

        # init parameters corectly
        self.reset_parameters()

        self.fn = RNNFunction.apply

    def forward(self, inputs: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [batch, sequence,
                input size].
            h0: initial hidden state.

        Returns:
            outputs tensor. Dimensions: [batch, sequence,
                hidden size].
            final hidden state for each element in the batch.
                Dimensions: [1, batch, hidden size].
        """

        return self.fn(
            inputs, h0, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh
        )

    def reset_parameters(self) -> None:
        """
        This method initializes the parameters in the correct way.
        """

        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

        return None