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
        grad_h = torch.empty(size=(N,L,Hout),device=h0.device,dtype=h0.dtype)
        grad_weight_ih = torch.empty(size=(Hout,L,Hin),device=weight_ih.device,dtype=weight_ih.dtype)
        grad_weight_hh = torch.empty(size=(Hout,L,Hout),device=weight_hh.device,dtype=weight_hh.dtype)
        grad_bias_ih = torch.empty(size=(L,Hout),device=bias_ih.device,dtype=bias_ih.dtype)
        grad_bias_hh = torch.empty(size=(L,Hout),device=bias_hh.device,dtype=bias_hh.dtype)

        