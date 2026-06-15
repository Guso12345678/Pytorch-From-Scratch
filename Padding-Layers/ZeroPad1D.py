import torch 
 
class ZeroPad1DFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs:torch.Tensor,padding_size:tuple): 
        """
        The input shape is: [N,C,Win]
        """
        N,C,Win = inputs.shape 
        pleft,pright = padding_size
        Wout = Win + pleft + pright
        outputs = torch.zeros((N,C,Wout))

        for i in range(Win): 
            outputs[:,:,i + pleft] = inputs[:,:,i]#para solo copiar los elementos desde el padding=0
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
