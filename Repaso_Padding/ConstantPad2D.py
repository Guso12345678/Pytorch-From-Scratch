import torch 

class ConstantPad2DFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs,padding_size,value): 
        """
            inputs of shape: [N,C,Hin,Win]
        """
        N,C,Hin,Win = inputs.shape 
        p_left,p_right,p_top,p_bottom = padding_size

        Hout = Hin + p_top + p_bottom
        Wout = Win + p_left + p_right 
        ctx.Hin = Hin 
        ctx.Win = Win 
        ctx.p_left = p_left
        ctx.p_top = p_top 
        outputs = torch.full(size=(N,C,Hout,Wout),fill_value=value)
        outputs[:,:,p_top:Hin+p_top,p_left:p_left+Win] = inputs 
        return outputs
    
    @staticmethod
    def backward(ctx,grad_outputs):
        """
            the output shape is: [N,C,Hout,Wout]
            the grad_inputs shape is: [N,C,Hin,Win]
        """
        H_in = ctx.Hin 
        W_in = ctx.Win 
        p_left = ctx.p_left 
        p_top = ctx.p_top 
        N,C,Hout = grad_outputs.shape 
        grad_inputs = torch.empty(size=(N,C,H_in,W_in))
        grad_inputs = grad_outputs[:,:,p_top:H_in+p_top,p_left:W_in+p_left]

        return grad_inputs