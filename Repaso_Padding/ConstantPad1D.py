import torch
class ConstantPad1DFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs,padding_size,value): 
        """
            inputs of shape: [N,C,Win]
            outputs of shape: [N,C,Wout]
        """
        N,C,Win = inputs.shape 
        p_left, p_right = padding_size
        Wout = Win + p_left + p_right 
        outputs = torch.full(size=(N,C,Wout),fill_value=value)

        ctx.Win = Win 
        ctx.p_left = p_left 
        outputs[:,:,p_left:Win+p_left] = inputs 
        return outputs 

    
    @staticmethod
    def backward(ctx,grad_outputs): 
        """
            grad_outputs of shape: [N,C,Wout]
        """
        Win = ctx.Win 
        p_left = ctx.p_left 
        N,C,Wout = grad_outputs.shape 
        grad_inputs = torch.empty(size=(N,C,Win))

        grad_inputs = grad_outputs[:,:,p_left:Win+p_left]
        return grad_inputs, None, None 
