import torch 

def softplus(inputs, beta): 
    return (1/beta)*torch.log(1 + torch.exp(beta*inputs)) 

def tanh(x): 
    return (torch.exp(x)-torch.exp(-x))/(torch.exp(x)+torch.exp(-x)) 

class MishFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs:torch.Tensor,beta): 
        ctx.save_for_backward(inputs)
        ctx.beta = beta
        return inputs* tanh(softplus(inputs,beta))
    @staticmethod
    def backward(ctx, grad_outputs): 
        inputs, = ctx.save_tensors 
        beta = ctx.beta 
        grad_inputs = tanh(softplus(inputs,beta)) + inputs*(1 - tanh(softplus(inputs,beta))**2)*(torch.exp(beta*inputs)/(1+torch.exp(beta*inputs)))
        return grad_inputs*grad_outputs, None  
