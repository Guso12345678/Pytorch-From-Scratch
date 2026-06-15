import torch 

class NLLLossFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs:torch.Tensor,targets:torch.Tensor):
        """
            inputs tiene un shape (N,C). 
            targets tiene un shape (N,)
            outputs como reduction igual a None sera (N,1)
        """
        ctx.save_for_backward(inputs,targets)
        N,_ = inputs.shape
        loss = torch.zeros((N,1))
        for i in range(N): 
            target_class = targets[i].item()
            loss[i,0] = -torch.log(inputs[i,target_class])
        return loss 

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors 
        N,_ = inputs.shape
        grad_inputs = torch.zeros_like(inputs)
        for i in range(N): 
            target_class = targets[i].item()
            grad_inputs[i,target_class] = -1/inputs[i,target_class]
        return grad_inputs, None         
