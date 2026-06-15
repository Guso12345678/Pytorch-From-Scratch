import torch 

class HardSigmoidFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)
        mask1 = inputs <= -3 
        mask2 = inputs >= 3 
        mask3 = (inputs > -3) & (inputs < 3)

        outputs = torch.zeros_like(inputs)
        outputs[mask1] = 0 
        outputs[mask2] = 1 
        outputs[mask3] = inputs[mask3]/6 + 0.5 
        
        return outputs 

    @staticmethod
    def backward(ctx,grad_outputs): 
        inputs, = ctx.saved_tensors
        mask1 = inputs <= -3 
        mask2 = inputs >= 3 
        mask3 = (inputs < -3) & (inputs > 3)
        grad_inputs = torch.zeros_like(grad_outputs)

        grad_inputs[mask1] = 0 
        grad_inputs[mask2] = 0 
        grad_inputs[mask3] = 1/6 

        return grad_inputs*grad_outputs


        