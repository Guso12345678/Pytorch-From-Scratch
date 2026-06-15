import torch 
class LayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape: tuple[int], eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(*normalized_shape))
            self.bias = torch.nn.Parameter(torch.empty(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(-len(self.normalized_shape),0))
        x_mean = torch.mean(x,dim=dims,keepdim=True)
        x_var = torch.var(x,dim=dims,keepdim=True,correction=0)

        x_norm = (x - x_mean)/torch.sqrt(x_var + self.eps)

        if self.affine: 
            shape =  [1]*(x.ndim - len(self.normalized_shape)) + list(self.normalized_shape) 
            weight_c = self.weight.view(*shape)
            bias_c = self.bias.view(*shape)
            x_norm = weight_c*x_norm + bias_c 
        return x_norm 
