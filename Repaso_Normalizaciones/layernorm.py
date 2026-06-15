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
        # Calcula media y varianza sobre las últimas `n` dimensiones
        dims = tuple(range(len(-self.normalized_shape),0))
        mean_c = torch.mean(x, dim=dims, keepdim=True) 
        var_c = torch.var(x, dim=dims, keepdim=True, correction=0)

        x_normalized = (x - mean_c)/(torch.sqrt(var_c + self.eps))

        if self.affine == True: 
            x_normalized = x_normalized*self.weight + self.bias 
        return x_normalized 











