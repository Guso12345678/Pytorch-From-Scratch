import torch 

class MultiMarginLossFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs:torch.Tensor,targets:torch.Tensor,p:int,margin:float): 
        N,C = inputs.shape 
        ctx.save_for_backward(inputs,targets)
        ctx.p = p 
        ctx.margin = margin 
        loss = torch.zeros(N,1)
        for i in range(N):
            target_class = targets[i].item()
            for j in range(C): 
                if j != target_class:
                    score_true = inputs[i,target_class]
                    score_otros = inputs[i,j]
                    diff = margin - score_true + score_otros 
                    if diff > 0: 
                        loss[i,0] += diff**p  
        return loss  
    
    @staticmethod
    def backward(ctx,grad_output): 
        inputs, targets = ctx.saved_tensors
        margin = ctx.margin 
        p = ctx.p 
        N,C = inputs.shape
        grad_inputs = torch.zeros_like(inputs)

        for i in range(N): 
            target_class = targets[i].item()
            for j in range(C): 
                if j != target_class: 
                    score_true = inputs[i,target_class]
                    score_otros = inputs[i,j]
                    diff = margin - score_true + score_otros 
                    if diff > 0: 
                        #Primero gradiente de cuando es correcto: 
                        grad_inputs[i,target_class] -= (p*(diff**(p-1)))
                        grad_inputs[i,j] += (p*(diff**(p-1)))
        return grad_inputs * grad_output, None, None, None 

        

