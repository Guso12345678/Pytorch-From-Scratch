import torch

def sigmoid(tensor): 
    return 1 / (1 + torch.exp(-tensor))
class MultiLabelSoftMarginLossFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, targets: torch.Tensor):  
        ctx.save_for_backward(inputs, targets)
        loss = targets*torch.log(sigmoid(inputs)) + (1 - targets)*torch.log(torch.exp(-inputs)*sigmoid(inputs))
        return -torch.mean(loss,dim=1)
    
    @staticmethod
    def backward(ctx,grad_outputs): 
        inputs, targets = ctx.saved_tensors 
        grad = (sigmoid(inputs) - targets) / inputs.shape[1]
        grad_outputs = grad_outputs.unsqueeze(1)
        return grad*grad_outputs, None 
    