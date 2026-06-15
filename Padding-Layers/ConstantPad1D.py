import torch 

class ConstantPad1DFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs,padding_size,value): 
        """
            inputs of shape: [N,C,Win]
            outputs of shape: [N,C,Wout]
        """
        N,C,Win = inputs.shape 
        pleft,pright = padding_size
        Wout = Win + pleft + pright
        outputs = torch.full((N,C,Wout),value)
        for i in range(Win): 
            outputs[:,:,i + pleft] = inputs[:,:,i] 
        ctx.save_for_backward(inputs)
        ctx.padding_size = padding_size
        return outputs 
    
    @staticmethod
    def backward(ctx,grad_outputs): 
        """
            the grad_outputs shape is: [N,C,Wout]
        """
        N,C,_ = grad_outputs.shape
        inputs, = ctx.saved_tensors 
        _,_,Win = inputs.shape 
        padding_size = ctx.padding_size 
        pleft,pright = padding_size
        grad_inputs = torch.empty((N,C,Win))
        for i in range(Win): 
            grad_inputs[:,:,i] = grad_outputs[:,:,i+pleft]
        #Hacer ese bucle seria equivalente a eso: 
        #grad_inputs = grad_outputs[:,:,pleft:pleft+Win]
        return grad_inputs,None