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

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[0], x.shape[1]
        x_prep = x.view(B, C, -1)

        if self.training or not self.track_running_stats:
            mean = x_prep.mean(dim=-1, keepdim=True)
            var = x_prep.var(dim=-1, keepdim=True, correction=0)
        else:
            # eval() mode with running stats
            mean = self.running_mean.view(1, C, 1).expand(B, C, 1)
            var = self.running_var.view(1, C, 1).expand(B, C, 1)

        x_norm = (x_prep - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            weight = self.weight.view(1, C, 1)
            bias = self.bias.view(1, C, 1)
            x_norm = weight * x_norm + bias

        return x_norm.view_as(x)
