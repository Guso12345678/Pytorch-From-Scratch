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
        maximo_inputs = torch.max(inputs,dim=1,keepdim=True)[0]
        inputs_stables = inputs - maximo_inputs 
        log_prob = torch.log(torch.sum(torch.exp(inputs_stables),dim=1,keepdim=True))
        log_prob_final = inputs_stables - log_prob 
        nll = -log_prob_final[torch.arange(inputs.shape[0]),targets]
        return nll 
    
    @staticmethod
    def backward(ctx:Any, grad_output:torch.Tensor):
        inputs,targets = ctx.saved_tensors  
        
        ##CALCULO DE LA PROB: 
        N, C = inputs.shape 
        maximo_inputs = torch.max(inputs,dim=1,keepdim=True)[0]
        inputs_stables = inputs - maximo_inputs
        probalidad = torch.exp(inputs_stables)/torch.sum(inputs_stables,dim=1,keepdim=True)

        ##CALCULO DEL BACKWARD: 
        grad_inputs = probalidad.clone()
        grad_inputs[torch.arange(N),targets] = grad_inputs[torch.arange(N),targets] -1 
        #Si hubiesemos aplicado media en el forward ahora tendriamos que nomralizar los grad_inputs por N 
        ##Multiplicamos por el grad_output: 
        grad_inputs = grad_inputs * grad_output
        return grad_inputs, None #porque las targets no tienen gradiente. 




