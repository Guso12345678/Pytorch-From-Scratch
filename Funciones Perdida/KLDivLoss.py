import torch 

class KLDivLossFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs:torch.Tensor,targets:torch.Tensor):
        ctx.save_for_backward(inputs,targets)
        outputs = inputs.clone()
        outputs = targets*(torch.log(targets) - torch.log(outputs))
        return outputs 
    @staticmethod
    def backward(ctx, grad_outputs):
         inputs,targets = ctx.saved_tensors
         grad_inputs = -targets/inputs 
         grad_inputs = grad_inputs * grad_outputs
         return grad_inputs, None 
