import torch 

class ReplicationPadFunction1D(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs,padding): 
        """
            The input shape is: [N,C,Win]
            the output shape is: [N,C,Wout]
        """
        N,C,Win = inputs.shape 
        pleft, pright = padding
        Wout = Win + pleft + pright 
        outputs = torch.zeros((N,C,Wout))

        outputs[:,:,:pleft] = torch.tensor(inputs[:,:,0]).unsqueeze(-1) #Sin el unsqueeze el tensor tiene un shape de [N,C] pero si ponemos un unsqueeze(-1) tiene un shape [N,C,1]
        outputs[:,:,pleft+Win:] = torch.tensor(inputs[:,:,-1]).unsqueeze(-1)

        outputs[:,:,pleft:pleft+Win] = inputs 

        ctx.padding_size = padding
        ctx.input_shape = inputs.shape
        return outputs 
     
    @staticmethod
    def backward(ctx, grad_outputs):
        pleft, pright = ctx.padding_size
        N, C, Win = ctx.input_shape

        grad_inputs = grad_outputs[:, :, pleft : pleft + Win] #Solo va fluir el gradiente por donde este ubicado el tensor original es decir desde el pleft hasta el Win + pleft. 
        return grad_inputs, None


    


