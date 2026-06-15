import torch 
class GroupNorm(torch.nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(num_channels))
            self.bias = torch.nn.Parameter(torch.empty(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,C = x.shape[0],x.shape[1]
        
        x_prep_for_grouped = x.view(B,self.num_groups,self.num_channels//self.num_groups,-1)
        
        dims = (-1,-2)
        
        mean = torch.mean(x_prep_for_grouped,dim=dims,keepdim=True)
        
        var = torch.var(x_prep_for_grouped,dim=dims,keepdim=True,correction=0)

        x_normalized = (x_prep_for_grouped - mean)/torch.sqrt(var + self.eps)
        
        x_final = x_normalized.view_as(x)
        
        if self.affine == True: 
            
            shape = [1,C] + [1]*(x_final.ndim - 2)
            
            weight = self.weight.view(*shape)
            
            bias = self.bias.view(*shape)

            x_final = weight*x_final + bias  
    
        return x_final
     
