import torch

class MaxPool2dEfficient(torch.autograd.Function):
    @staticmethod
    def forward(ctx,inputs:torch.Tensor,padding,kernel_size,stride,dilation): 
        """
            the shape of the inputs is: [N,C,H,W]
            the shape of the desired outputs is: [N,C,Hout,Wout]
        """
        N,C,H,W = inputs.shape 
        
        inputs_unfoled = inputs.unfold(2,size=kernel_size[0],step=stride[0]).unfold(3,size=kernel_size[1],step=stride[1]) #Es primero haces el unfold por anchura y luego haces el unfold por altura, es decir descomprimir primero por anchura y luego por altura.   
        Hout, Wout = inputs_unfoled.shape[2], inputs_unfoled.shape[3]
        inputs_finales = inputs_unfoled.reshape(N,C,Hout,Wout,-1) #Ahora sera de la forma de [N,C,Hout,Wout,K(Kernel_size*Kernel_size)]
        output, max_idx = inputs_finales.max(dim=-1)
        ctx.save_for_backward(max_idx)
        ctx.kernel_size = kernel_size
        ctx.H = H 
        ctx.W = W
        return output
    def backward(ctx,grad_outputs:torch.Tensor):
        """
            the shape of grad_outputs is : [N,C,Hout,Wout]
        """ 
        max_idx, = ctx.saved_tensors
        kernel_size = ctx.kernel_size 
        H = ctx.H 
        W = ctx.W
        Kh,Kw = kernel_size
        N,C,H_out,W_out = grad_outputs.shape 
        grad_expanded = torch.zeros((N,C,H_out,W_out,Kh*Kw))
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        k = max_idx[n, c, i, j].item()
                        grad_expanded[n, c, i, j, k] = grad_outputs[n, c, i, j]#Esto nos da un shape de: [N,C,H_out,W_out,K]
        grad_expanded_permuted = grad_expanded.permute(0,1,4,2,3).reshape(N,C*Kh*Kw,H_out*W_out)
        grad_inputs = torch.nn.functional.fold(grad_expanded_permuted,output_size=(H,W),kernel_size=(Kh,Kw),stride=stride)
        return grad_inputs, None, None, None  


        







# ======= TEST CASE =======
torch.manual_seed(0)

# Input: batch de 1, 1 canal, 4x4 imagen
x = torch.arange(1., 17.).reshape(1, 1, 4, 4)
print("Input:\n", x)

# Parámetros del pool
kernel_size = (2, 2)
stride = (1, 1)
padding = (0, 0)
dilation = (1, 1)

# Aplica tu función
output = MaxPool2dEfficient.apply(x, padding, kernel_size, stride, dilation)
print("\nOutput (unfolded):\n", output)