import torch 

class Fold1DFunctional(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs:torch.Tensor,dilation,stride,kernel_size):
        """
            El tensor de entrada tendra la forma de [N,C*kernel_size,L_out], donde L_out son las ventanas que caben. 
            Vamos a suponer que no hay padding.         
        """ 
        N,filas,L_out = inputs.shape
        C = filas // kernel_size
        ctx.save_for_backward(torch.tensor(L_out))
        ctx.kernel_size = kernel_size
        ctx.stride = stride 
        ctx.dilation = dilation 
        ctx.C = C 
        L_in = int(((L_out + 2*0 - dilation*(kernel_size -1) - 1)/(stride)) + 1)
        output = torch.empty((N,C,L_in))

        for l in range(L_out): 
            for k in range(kernel_size): 
                start = l*stride + k*dilation
                output[:,:,start] += inputs[:,k::kernel_size,l]
        return output #Ahora nuestro tensor tiene el shape: [N,C,L_in] 
    

    @staticmethod
    def backward(ctx,grad_outputs:torch.Tensor): 
        """
            El grad_outputs tiene que tener el mismo shape que los outputs por lo cual es: [N,C,L_in] y tendremos que llevar a cabo ahora la operacion de unfold para obtener: [N,C*kernel_size,L_out] que es el shape que tienen los inputs
        """
        L_out, = ctx.save_tensors
        L_out = L_out.item()
        kernel_size = ctx.kernel_size 
        stride = ctx.stride 
        dilation = ctx.dilation 
        C = ctx.C 
        grad_inputs = torch.empty((grad_outputs.size(0),C*kernel_size,L_out))
        for l in range(L_out): 
            for k in range(kernel_size): 
                start = l*stride + k*dilation
                grad_inputs[:,k::kernel_size,l] = grad_outputs[:,:,start]
        return grad_inputs,None, None, None, None   