import torch 

class SoftMarginLossFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs,targets): 
        ctx.save_for_backward(inputs,targets)
        loss = torch.log(1 + torch.exp(-targets*inputs))
        return loss 
    @staticmethod 
    def backward(ctx,grad_outputs): 
        inputs,targets = ctx.saved_tensors 
        grad_inputs = (-targets*torch.exp(-inputs*targets))/(1 + torch.exp(-inputs*targets))
        return grad_inputs*grad_outputs, None 
    