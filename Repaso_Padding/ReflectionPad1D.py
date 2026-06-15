import torch 

class ReflectionPad1DFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs,padding_size): 
        """
            the inputs shape is: [N,C,Win]
            the output shape is: [N,C,Wout]
        """
        N,C,Win = inputs.shape 
        p_left,p_right = padding_size 
        Wout = Win + p_left + p_right 
        outputs = torch.empty(size=(N,C,Wout))
        outputs[:,:,:p_left] = torch.flip(inputs[:,:,1 : p_left+1],dim=[-1])
        outputs[:,:,p_left+Win:] = torch.flip(inputs[:,:,-p_right-1 : -1],dim=[-1])
        outputs[:,:,p_left:p_left+Win] = inputs 
        return outputs 