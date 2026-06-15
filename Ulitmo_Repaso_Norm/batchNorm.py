import torch


class BatchNorm(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        This is the constructor of the BatchNorm class.

        Args:
            num_features: number of input channels/features.
            eps: epsilon to avoid division by zero. Defaults to 1e-5.
            momentum: running stats momentum. Defaults to 0.1.
            affine: whether to learn γ and β. Defaults to True.
        """
        super().__init__()


        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.num_features = num_features

        # Create learnable parameters if affine=True
        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(num_features, dtype=dtype))
            self.bias = torch.nn.Parameter(torch.empty(num_features, dtype=dtype))

        # Running statistics (no grad)
        self.register_buffer("running_mean", torch.zeros(num_features, dtype=dtype))
        self.register_buffer("running_var", torch.ones(num_features, dtype=dtype))

        self.reset_parameters()
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass of the BatchNorm module.

        Args:
            inputs: input tensor. Dimensions: [batch, channels, *].

        Returns:
            outputs tensor. Dimensions: [batch, channels, *].
        """
        B,C = inputs.shape[0],inputs.shape[1]
        inputs_prep = inputs.clone().view(B,C,-1)
        
        if self.training: 
            mean_c = torch.mean(inputs_prep,dim=(0,-1),keepdim=True) #[1,C,1]
            var_c = torch.var(inputs_prep,dim=(0,-1),keepdim=True,correction=0)#[1,C,1]

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_c.squeeze(0).squeeze(2)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_c.squeeze(0).squeeze(2)

        else: 
            mean_c = self.running_mean.unsqueeze(0).unsqueeze(2)
            var_c = self.running_var.unsqueeze(0).unsqueeze(2)
        
        inputs_norm = (inputs_prep - mean_c)/torch.sqrt(var_c + self.eps)
        if self.affine: 
            weight_c = self.weight.unsqueeze(0).unsqueeze(2)
            bias_c = self.bias.unsqueeze(0).unsqueeze(2)
            inputs_norm = weight_c*inputs_norm + bias_c 
        return inputs_norm.view_as(inputs)

        