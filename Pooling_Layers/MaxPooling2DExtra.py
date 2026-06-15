# deep learning libraries
import torch
import torch.nn.functional as F

# other libraries
from typing import Optional, Any


def unfold_max_pool_2d(
    inputs: torch.Tensor, kernel_size: int, stride: int, padding: int
) -> torch.Tensor:
    """
    This function computes the unfold needed for the MaxPool2d.
    Since the maxpool only computes the max over single channel
    and not over all the channels, we need that the second dimension of
    our unfold tensors are data from only channel. For that, we will
    include the channels into another dimension that will
    not be affected by the consequently operations.

    Args:
        inputs: inputs tensor. Dimensions: [batch, channels, height,
            width].
        kernel_size: size of the kernel to use. In this case the
            kernel will be symmetric, that is why only an integer is
            accepted.
        stride: stride to use in the maxpool operation. As in the case
            of the kernel size, the stride willm be symmetric.
        padding: padding to use in the maxpool operation. As in the
            case of the kernel.

    Returns:
        inputs unfolded. Dimensions: [batch * channels,
            kernel size * kernel size, number of windows].
    """

    # TODO
    B,C,_,_ = inputs.shape
    # inputs_prep = inputs.view(B*C,H,W)
    inputs_unfolded = torch.nn.functional.unfold(inputs, kernel_size=(kernel_size,kernel_size), stride=(stride,stride), padding=(padding,padding))#El unfold te devuelve por defecto [B,C*kernel*kernel,num_windows]
    B,dim_rara,num_windows = inputs_unfolded.shape
    inputs_finales = inputs_unfolded.view(B*C,dim_rara//C,num_windows)

    return inputs_finales


def fold_max_pool_2d(
    inputs: torch.Tensor,
    output_size: int,
    batch_size: int,
    kernel_size,
    stride: int,
    padding: int,
) -> torch.Tensor:
    """
    This function computes the fold needed for the MaxPool2d.
    Since the maxpool only comute sthe max over single channel
    and not over all the channels, we need that the second dimension of
    our unfold tensors are data from only channel. To do that, we
    this fold version recovers the channel dimensions before executing 
    the fold operation.

    Args:
        inputs: inputs unfolded. Dimensions: [batch * channels,
            kernel size * kernel size, number of windows].
        output_size: output size for the fold, i.e., the height and
            the width.
        batch_size: batch size
        stride: stride to use in the maxpool operation. As in the case
            of the kernel size, the stride willm be symmetric.
        padding: padding to use in the maxpool operation. As in the
            case of the kernel.

    Returns:
        inputs folded. Dimensions: [batch, channels, height, width].
    """

    # TODO
    ByC, K, num_windows = inputs.shape 
    C = ByC // batch_size
    inputs_prep_for_fold = inputs.view(batch_size,C*K,num_windows)
    inputs_foldeados = torch.nn.functional.fold(inputs_prep_for_fold,output_size=(output_size,output_size),kernel_size=(kernel_size,kernel_size),stride=(stride,stride),padding=(padding,padding))
    return inputs_foldeados

class MaxPool2dFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the MaxPool2d.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> torch.Tensor:
        """
        This is the forward method of the MaxPool2d.

        Args:
            ctx: context for saving elements for the backward.
            inputs: inputs for the model. Dimensions: [batch,
                channels, height, width].

        Returns:
            output of the layer. Dimensions:
                [batch, channels,
                (height + 2*padding - kernel size) / stride + 1,
                (width + 2*padding - kernel size) / stride + 1]
        """
        B,C,H,W = inputs.shape 
        inputs_polleados = unfold_max_pool_2d(inputs,kernel_size,stride,padding)
        ByC,K,num_windows = inputs_polleados.shape 
        inputs_polleados2 = inputs_polleados.view(B,C,K,num_windows)
        Hout = int(torch.sqrt(torch.tensor(num_windows))) 

        inputs_polled = inputs_polleados2.transpose(2,3).view(B,C,Hout,Hout,K)
        inputs_finales,max_idx = inputs_polled.max(dim=-1,keepdim=True)
        
        inputs_polled_cambiados = inputs_polled.view(B,C,Hout*Hout,K).transpose(2,3)
        mask = torch.zeros_like(inputs_polled_cambiados,dtype=torch.bool).scatter(2,max_idx.permute(0,1,4,2,3).view(B,C,1,-1),True) #Vamos a obtener una mascara de shape: [B,C,K*K,Hout*Hout]
        ctx.save_for_backward(mask)
        ctx.H = H 
        ctx.W = W 
        ctx.kernel_size = kernel_size
        ctx.stride = stride 
        ctx.padding = padding 
        return inputs_finales.squeeze(-1) 

        




    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        """
        This method is the backward of the MaxPool2d.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions:
                [batch, channels,
                (height + 2*padding - kernel size) / stride + 1,
                (width + 2*padding - kernel size) / stride + 1]

        Returns:
            inputs gradients dimensions: [batch, channels,
                height, width].
            None value.
            None value.
            None value.
        """

        # TODO
        mask, = ctx.saved_tensors
        B,C,Hout,Wout = grad_outputs.shape 
        kernel_size = ctx.kernel_size 
        stride = ctx.stride 
        padding = ctx.padding  
        H = ctx.H 
        W = ctx.W 


        grad_outputs_expanded = grad_outputs.unsqueeze(2).expand(B,C,kernel_size*kernel_size,Hout,Wout)#El unsqueeze es necesario para poder aplicar el expand donde nuestras dimensiones seran: [N,C,K*K,Hout,Wout]. 
        #Para poder multiplicarlo por la mascara que contiene donde se encuentran ubicados los valores maximos tenemos que hacer alguna transformacion: la mascara es de shape: [B,C,K*K,Hout*Hout] 
        grad_outputs_expanded_final = grad_outputs_expanded.view(B,C,kernel_size*kernel_size,Hout*Wout) #Ahora obtenemos un tensor de shape: [N,C,K*K,Hout*Hout]

        grad_tras_mascara = grad_outputs_expanded_final * mask #shape: [N,C,K*K,Hout*Hout] 

        grad_inputs = grad_tras_mascara.view(B*C,kernel_size*kernel_size,Hout*Hout) #Shape [N*C,K*K,number_windows] perfecto para el Fold. 

        grad_inputs_final = fold_max_pool_2d(grad_inputs,output_size=H,batch_size=B,kernel_size=kernel_size,stride=stride,padding=padding)

        return grad_inputs_final, None, None, None 


        





class MaxPool2d(torch.nn.Module):
    """
    This is the class that represents the MaxPool2d Layer.
    """

    kernel_size: int
    stride: int

    def __init__(
        self, kernel_size: int, stride: Optional[int], padding: int = 0
    ) -> None:
        """
        This method is the constructor of the MaxPool2d layer.
        """

        # call super class constructor
        super().__init__()

        # set attributes value
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding

        # save function
        self.fn = MaxPool2dFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [batch, channels,
                output channels, height, width].

        Returns:
            outputs tensor. Dimensions: [batch, channels,
                height - kernel size + 1, width - kernel size + 1].
        """

        return self.fn(inputs, self.kernel_size, self.stride, self.padding)
