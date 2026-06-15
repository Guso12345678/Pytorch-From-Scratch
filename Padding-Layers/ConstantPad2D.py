import torch 

class ConstantPad2DFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs,padding_size,value): 
        """
            inputs of shape: [N,C,Hin,Win]
        """
        N,C,Hin,Win = inputs.shape 
        pleft,pright,ptop,pbottom = padding_size
        Wout = Hin + ptop + pbottom
        Hout = Win + pleft + pright 
        outputs = torch.full((N,C,Wout,Hout),value)
        ctx.save_for_backward(inputs)
        ctx.padding_size = padding_size
        for i in range(Hin): 
            for j in range(Win): 
                outputs[:,:,i+ptop , j+pleft] = inputs[:,:,i,j]
        return outputs
    
    @staticmethod
    def backward(ctx,grad_outputs):
        """
            the output shape is: [N,C,Hout,Wout]
            the grad_inputs shape is: [N,C,Hin,Win]
        """ 
        inputs, = ctx.saved_tensors
        padding_size = ctx.padding_size 
        pleft,pright,ptop,pbottom = padding_size
        N,C,Hout,Wout = grad_outputs.shape 
        N,C,Hin,Win = inputs.shape 
        grad_inputs = torch.empty((N,C,Hin,Win))
        ##Primera forma usando bucles: 
        for i in range(Hin): 
            for j in range(Win): 
                grad_inputs[:,:,i,j] = grad_outputs[:,:,i+ptop,j+pleft]
        #Segunda forma rellenado automatico: 
        #grad_inputs = grad_outputs[:,:,ptop:ptop+Hin,pleft:pleft+Win]
        return grad_inputs,None


