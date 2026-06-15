import torch 
 
class ZeroPad1DFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs:torch.Tensor,padding_size:tuple): 
        """
        The input shape is: [N,C,Win]
        The outputs shape is: [N,C,Wout]
        """
        N,C,Win = inputs.shape 
        p_left,p_right = padding_size 
        Wout = Win + p_left + p_right 

        outputs = torch.zeros(size=(N,C,Wout),dtype=inputs.dtype,device=inputs.device)
        outputs[:,:,p_left:p_left+Win] = inputs
        ctx.Win = Win
        ctx.p_left = p_left 
        ctx.p_right = p_right  
        return outputs
    @staticmethod
    def backward(ctx,grad_outputs): 
        """
            the grad_outputs shape is: [N,C,Wout]
        """
        N,C,_ = grad_outputs.shape 
        Win = ctx.Win 
        p_left = ctx.p_left 
        p_right = ctx.p_right 
        grad_inputs = torch.empty(size=(N,C,Win))

        grad_inputs[:,:,:] = grad_outputs[:,:,p_left:Win+p_left]
        return grad_inputs, None 


import torch

# Suponemos que ya tienes la clase definida como ZeroPad1DFunction

# Parámetros
N, C, Win = 2, 3, 4
p_left, p_right = 1, 2
padding_size = (p_left, p_right)

# Input de prueba con requires_grad
inputs = torch.randn(N, C, Win, requires_grad=True)

# Forward
padded_output = ZeroPad1DFunction.apply(inputs, padding_size)
print("Padded output shape:", padded_output.shape)  # Esperado: [N, C, Win + p_left + p_right]

# Probar backward
loss = padded_output.sum()
loss.backward()

print("Gradientes de los inputs:")
print(inputs.grad)  # Debería tener el mismo shape que inputs: [N, C, Win]

