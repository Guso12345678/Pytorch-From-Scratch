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
        super()._init_()

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

        # TODO: implement batch normalization
        B,C,_ = inputs.shape 
        inputs_preparados = inputs.clone().view(B,C,-1) #Se calcula la media y todo solo para los canales: 
        
        if self.training: 
            media_c = torch.mean(inputs_preparados,dim=(0,2),keepdim=True)
            var_c = torch.var(inputs_preparados,dim=(0,2),keepdim=True,correction=0)

            self.running_mean = self.momentum*self.running_mean + (1 - self.momentum)*media_c.squeeze(2).squeeze(0)
            self.running_var = self.momentum*self.running_var + (1 - self.momentum)*var_c.squeeze(2).squeeze(0)
        else: 
            media_c = self.running_mean.view(1,C,1)
            var_c = self.running_var.view(1,C,1)
        inputs_normalized = (inputs_preparados - media_c)/torch.sqrt(var_c + self.eps)
        if self.affine: 
            weights_preparados = self.weight.unsqueeze(0).unsqueeze(2)
            bias_preparados = self.bias.unsqueeze(0).unsqueeze(2)
            inputs_normalized = weights_preparados*inputs_normalized + bias_preparados
        return inputs_normalized.view_as(inputs)


