import torch 
from typing import Any

class CrossEntropyLossFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx:Any, inputs:torch.Tensor, targets:torch.Tensor): 
        """
        La entrada va a tener un shape de: (N,C), logits
        Los targets va a tener un shape de: (N,)
        """
        ctx.save_for_backward(inputs,targets)
        maximo_inputs,_ = torch.max(inputs,dim=1,keepdim=True)
        inputs_estables = inputs - maximo_inputs 

        log = inputs_estables - torch.log(torch.sum(torch.exp(inputs_estables),dim=1,keepdim=True))
        log_prob_final = inputs_estables - log 
        outputs = -log_prob_final[torch.arange(inputs.shape[0]),targets]
        return outputs 

    
    @staticmethod
    def backward(ctx,grad_outputs): 
        inputs, targets = ctx.saved_tensors 
        N,C = inputs.shape 

        maximo_inputs = torch.max(inputs,dim=1,keepdim=True)[0]
        inputs_estables = inputs - maximo_inputs 
        prob = torch.exp(inputs_estables)/torch.sum(torch.exp(inputs_estables),dim=1,keepdim=True)

        ##CALCULO DEL BACKWARD 
        grad_inputs = prob.clone()
        grad_inputs[torch.arange(N),targets] = prob[torch.arange(N),targets] - 1 
        return grad_inputs*grad_outputs,None 
