import torch 

class ReflectionPad1DFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs,padding_size): 
        """
            the inputs shape is: [N,C,Win]
            the output shape is: [N,C,Wout]
        """
        N,C,Win = inputs.shape 
        pleft, pright = padding_size
        Wout = Win + pleft + pright 

        outputs = torch.zeros((N,C,Wout))

        outputs[:,:,:pleft] = torch.flip(inputs[:,:,0:pleft],dims=[-1])
        outputs[:, :, pleft+Win:] = torch.flip(inputs[:, :, -pright:], dims=[-1])
        outputs[:,:,pleft:pleft+Win] = inputs 
        ctx.padding_size = padding_size
        ctx.input_shape = inputs.shape
        return outputs 
    
    @staticmethod
    def backward(ctx, grad_outputs):
        pleft, pright = ctx.padding_size
        _, _, Win = ctx.input_shape
        grad_inputs = grad_outputs[:, :, pleft:pleft+Win]
        return grad_inputs, None
