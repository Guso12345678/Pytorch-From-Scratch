import torch 

class CircularPad2DFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs,padding_size): 
        """
            the inputs shape is: [N,C,Hin,Win]
            the outputs shape is: [N,C,Hout,Wout]
        """
        N,C,Hin,Win = inputs.shape 
        pleft,pright,ptop,pbottom = padding_size 
        Hout = Hin + ptop + pbottom
        Wout = Win + pleft + pright 

        outputs = torch.zeros((N,C,Hout,Wout))

        #Primero vamos a rellenarlo por la izquierda y derecha:  
        outputs[:,:,:,:pleft] = inputs[:,:,:,-pleft:]
        outputs[:,:,:,pleft + Win:] = inputs[:,:,:,:pright]

        #Ahora vamos a rellenar por arriba y por debajo: 
        outputs[:,:,:ptop,:] = inputs[:,:,-ptop:,:]
        outputs[:,:,ptop+Hin:,:] = inputs[:,:,:pbottom,:]

        #Ahora llenamos la matriz del medio: 
        outputs[:,:,ptop:ptop+Hin, pleft:pleft+Win] = inputs

        return outputs  
    

    @staticmethod
    def backward(ctx, grad_outputs):
        pleft, pright, ptop, pbottom = ctx.padding_size
        _, _, Hin, Win = ctx.input_shape
        grad_inputs = grad_outputs[:, :, ptop:ptop+Hin, pleft:pleft+Win]
        return grad_inputs, None
