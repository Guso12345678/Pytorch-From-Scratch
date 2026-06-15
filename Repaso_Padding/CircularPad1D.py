import torch 
class CircularPadding1DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, padding_size):
        """
        inputs: shape [N, C, Win]
        padding_size: (pad_left, pad_right)
        """
        N,C,Win = inputs.shape 
        p_left, p_right = padding_size
        Wout = Win + p_left + p_right 
        outputs = torch.empty(size=(N,C,Wout))
        outputs[:,:,:p_left] = inputs[:,:,-p_left:]
        outputs[:,:,Win + p_left:] = inputs[:,:,:p_right]
        outputs[:,:,p_left:Win+p_left] = inputs 
        ctx.padding_size = padding_size
        ctx.input_shape = inputs.shape
        return outputs
    
    @staticmethod
    def backward(ctx, grad_outputs):
        pleft, pright = ctx.padding_size
        N, C, Win = ctx.input_shape

        grad_inputs = grad_outputs[:, :, pleft : pleft + Win] #Solo va fluir el gradiente por donde este ubicado el tensor original es decir desde el pleft hasta el Win + pleft. 
        return grad_inputs, None 