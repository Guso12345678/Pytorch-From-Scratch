import torch 


def tanh(x): 
    return (torch.exp(x)-torch.exp(-x))/(torch.exp(x)+torch.exp(-x)) 

class GELUFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs): 
        ctx.save_for_backward(inputs)
        dentro = ((2/torch.pi)**0.5)*(inputs + 0.044715*inputs**3)
        outputs = 0.5*inputs*( 1 + tanh(dentro))
        return outputs 
    
    @staticmethod
    def backward(ctx,grad_outputs): 
        inputs, = ctx.saved_tensors
        dentro = ((2/torch.pi)**0.5)*(inputs + 0.044715*inputs**3)
        derivada_dentro = ((2/torch.pi)**0.5) + ((2/torch.pi)**0.5)*0.044715*3*inputs**2
        grad_inputs = (0.5 + 0.5*tanh(dentro)) + (0.5*inputs)*(1 - tanh(dentro)**2)*derivada_dentro
        return grad_inputs * grad_outputs 