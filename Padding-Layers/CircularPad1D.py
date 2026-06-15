import torch 
class CircularPadding1DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, padding_size):
        """
        inputs: shape [N, C, Win]
        padding_size: (pad_left, pad_right)
        """
        N, C, Win = inputs.shape
        pleft, pright = padding_size
        Wout = Win + pleft + pright
        outputs = torch.empty((N, C, Wout), device=inputs.device, dtype=inputs.dtype)

        #Primero rellenamos la parte de la izquierda del tensor: 
        outputs[:,:,:pleft] = inputs[:,:,-pleft:]

        #Ahora tenemos que rellenar como tal con el tensor de inputs: 
        outputs[:,:,pleft:Win+pleft] = inputs
        
        #Ahora tenemos que rellenar la parte de la derecha del tensor para ello usaremos los primmeros elementos en funcion del pright
        outputs[:,:,pleft + Win:] = inputs[:,:,:pright]        

        ctx.padding_size = padding_size
        ctx.input_shape = inputs.shape
        return outputs 

    @staticmethod
    def backward(ctx, grad_outputs):
        pleft, pright = ctx.padding_size
        N, C, Win = ctx.input_shape

        grad_inputs = grad_outputs[:, :, pleft : pleft + Win] #Solo va fluir el gradiente por donde este ubicado el tensor original es decir desde el pleft hasta el Win + pleft. 
        return grad_inputs, None