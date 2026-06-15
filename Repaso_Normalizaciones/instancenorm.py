import torch 
class InstanceNorm(torch.nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True, track_running_stats: bool = False) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(num_features))
            self.bias = torch.nn.Parameter(torch.empty(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,C = x.shape[0],x.shape[1] 
        dims = tuple(range(2,x.ndim))

        if self.training:
            mean = x.mean(dim=dims, keepdim=True)
            var = x.var(dim=dims, keepdim=True, correction=0)

            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = torch.mean(mean,dim=0).view_as(self.running_mean) #shape: [num_features]
                    self.running_var = torch.var(var,dim=0).view_as(self.running_var)
                    self.num_batches_tracked += 1 
        else: 
            if self.track_running_stats == True: 
                mean = self.running_mean.view(1,C,*([1] * (x.ndim - 2 ))) #tiene que ser shape: [1,C,tantos 1s como sea necesario], es decir el numero de dimensiones
                var = self.running_var.view(1,C,*([1] * (x.ndim - 2 )))
            else: 
                mean = x.mean(dim=dims, keepdim=True)
                var = x.var(dim=dims, keepdim=True, correction=0)

        x_normalized = (x - mean)/torch.sqrt(var + self.eps) 
        if self.affine == True: 
            weights_reshaped = self.weight.view(1,C,*([1]*(x.ndim - 2)))
            bias_reshaped = self.bias.view(1,C,*([1]*(x.ndim -2)))
            x_normalized = weights_reshaped*x_normalized + bias_reshaped
        return x_normalized





        
