import torch

def sigmoid(tensor): 
    return 1 / (1 + torch.exp(-tensor))

class MultiLabelSoftMarginLossFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, targets: torch.Tensor):
        ctx.save_for_backward(inputs, targets)
        loss = -((targets * torch.log(sigmoid(inputs))) + 
                 ((1 - targets) * torch.log(1 - sigmoid(inputs))))
        return torch.mean(loss, dim=1)  # shape (N,)

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad = (sigmoid(inputs) - targets) / inputs.shape[1]  # shape (N, C), demostracion de la derivada a papel. 
        # grad_outputs shape: (N,) → expand to (N, C)
        grad_outputs = grad_outputs.unsqueeze(1)  # (N, 1)
        return grad * grad_outputs, None
