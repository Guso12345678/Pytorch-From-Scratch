import torch 
class CrossEntropyLoss(nn.Module):
    """Cross Entropy Loss"""

    def _init_(self, reduction):
        super()._init_()
        self.reduction = reduction

    def forward(self, ctx, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass (loss computation)
        input_stable = input - input.max(dim=1, keepdim=True)[0]
        exp_input = torch.exp(input_stable)
        sum_exp = exp_input.sum(dim=1, keepdim=True)
        softmax = exp_input / sum_exp

        ctx.save_for_backward(input, target, softmax)
        ctx.N = input.shape[0]

        loss = -torch.sum(target * torch.log(softmax), dim=1)

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)

        return loss

    def backward(self, ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # TODO: Implement backward pass (manual gradient computation)
        input, target, softmax = ctx.saved_tensors
        N = ctx.N

        grad = softmax - target

        if self.reduction == "mean":
            grad = grad / N

        grad_input = grad_output * grad
        return grad_input