# deep learning libraries
import torch


class GroupNorm(torch.nn.Module):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        This is the constructor of the GroupNorm class.

        Args:
            num_groups: number of groups to use.
            num_channels: number of channels to use.
            eps: epsilon to avoid overflow. Defaults to 1e-5.
            affine: Indicator to perform affine transformation.
                Defaults to True.
        """

        # call super class constructor
        super().__init__()

        # save attributes
        self.num_groups = num_groups
        self.eps = eps
        self.affine = affine

        # create parameters if it is an affine transformation
        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(num_channels, dtype=dtype))
            self.bias = torch.nn.Parameter(torch.empty(num_channels, dtype=dtype))

        self.reset_parameters()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass of the module.

        Args:
            inputs: input tensor. Dimensions: [batch, channels, *].

        Returns:
            outputs tensor. Dimensions: [batch, channels, *].
        """

        # TODO
        B,C = inputs.shape[0],inputs.shape[1]
        inputs_prep = inputs.clone().reshape(B,self.num_groups,C//self.num_groups,-1)
        mean_c = torch.mean(inputs_prep,dim=(-1,-2),keepdim=True)
        var_c= torch.var(inputs_prep,dim=(-1,-2),keepdim=True,correction=0)
        inputs_norm = (inputs_prep - mean_c)/torch.sqrt(var_c + self.eps)
        inputs_final = inputs_norm.view_as(inputs)
        if self.affine == True: 
            shape = [1,C] + [1]*(inputs.ndim - 2)
            weight_c = self.weight.view(*shape)
            bias_c = self.weight.view(*shape)
            inputs_norm = inputs_final*weight_c + bias_c 
        
        return inputs_final