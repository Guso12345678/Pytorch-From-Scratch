import torch
class MarginRankingLossFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs1,inputs2,targets,margin): 
        ctx.save_for_backward(inputs1,inputs2,targets)
        ctx.margin = margin 
        outputs = torch.zeros_like(inputs1)
        mask1 = -targets*(inputs1 - inputs2) + margin > 0 
        mask2 = -targets*(inputs1 - inputs2) + margin <= 0
        outputs[mask1] = -targets[mask1]*(inputs1[mask1] - inputs2[mask1]) + margin 
        outputs[mask2] = 0  
        return outputs 

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs1, inputs2, targets = ctx.saved_tensors
        margin = ctx.margin 
        mask1 = -targets*(inputs1 - inputs2) + margin > 0 
        mask2 = -targets*(inputs1 - inputs2) + margin <= 0 
        grad_inputs1 = grad_outputs.clone()
        grad_inputs2 = grad_outputs.clone()
        grad_inputs1[mask1] *= -targets[mask1]
        grad_inputs1[mask2] *= 0 
        grad_inputs2[mask1] *= targets[mask1]
        grad_inputs2[mask2] *= 0
        return grad_inputs1,grad_inputs2, None, None  