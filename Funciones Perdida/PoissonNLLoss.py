import torch 

class PoissonNLLossFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs:torch.Tensor,targets:torch.Tensor,log_input:bool,full:bool, eps:float): 
        ctx.save_for_backward(inputs,targets)
        ctx.log_input = log_input
        ctx.eps = eps
        stirling_approximation = targets*torch.log(targets) - targets + 0.5*torch.log(2*torch.pi*targets)
        if log_input == True: 
            loss = torch.exp(inputs) - targets*inputs + stirling_approximation if full == True else torch.exp(inputs) - targets*inputs
        else: 
            loss = inputs - targets*torch.log(inputs + eps) + stirling_approximation if full == True else inputs - targets*torch.log(inputs + eps)
        return loss 
    
    @staticmethod
    def backward(ctx,grad_outputs): 
        inputs,targets = ctx.saved_tensors
        log_inputs = ctx.log_input 
        eps = ctx.eps 
        grad_inputs = torch.exp(inputs) - targets if log_inputs == True else torch.ones(inputs.shape) - (targets)/(inputs + eps) 
        return grad_inputs*grad_outputs, None, None, None