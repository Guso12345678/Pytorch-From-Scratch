import torch
from typing import Any
class Conv1DFunctional(torch.autograd.Function): 
    @staticmethod
    def forward(ctx:Any,inputs:torch.Tensor,kernel:torch.Tensor): 
        """
            the shape of the inputs will be: [N,Cin,Lin]
            the shape of the kernel will be: [Cout,Cin,kernel_size]
            the shape of the output will be: [N,Cout,Lout]
        """ 
        #Hacemos primero la operacion de unfold para luego poder multiplicar:
        N, C_in, L_in = inputs.shape 
        ctx.L_in = L_in
        Cout,C_in,K = kernel.shape 
        L_out = L_in - K + 1 
        
        inputs_unfolded = inputs.unfold(dimension=2,size=K,step=1) #La salida que obtenemos ahora es: (N,C_in,Lout,K)
        ctx.save_for_backward(inputs_unfolded,kernel)
        kernel_shapeado = kernel.view(Cout,-1) #Esto va a tener shape de: [Cout,C_in*K]  
        
        
        inputs_unfolded = inputs_unfolded.permute(0,2,1,3).reshape(N,L_out,C_in*K) 
        

        outputs = inputs_unfolded @ kernel_shapeado.T #Porque aqui tenemos: [N,Lout,Cout]
        return outputs.transpose(1,2) #Se le da la vuelta para nos quede [N,Cout,Lout]

    @staticmethod
    def backward(ctx,grad_outputs:torch.Tensor): 
        """
            El gradiente de los outputs es: [N,Cout,Lout]
        """
        inputs,kernel = ctx.saved_tensors #Donde sabemos que el kernel es de shape: [Cout,Cin,K] y los inputs son: [N,Cin,Lout,K]
        #Primero vamos a hacer el gradiente de los pesos: que queremos el shape de
        L_in = ctx.L_in
        N,Cout,Lout = grad_outputs.shape
        N,Cin,Lout,K = inputs.shape  
        grad_outputs = grad_outputs.clone().permute(1,0,2).reshape(Cout,N*Lout)#Tenemos un shape de: [Cout,N*Lout]
        inputs_preparados = inputs.permute(0, 2, 1, 3).reshape(-1, Cin * K) # (N * Lout, Cin * K)
        #Gradiente de los pesos: 
        grad_weights = grad_outputs.T @ inputs_preparados #(Cout, Cin * K)
        grad_weights = grad_weights.view(Cout,Cin,K)

        #Gradiente de los inputs tiene que ser de shape: [N,Cin,Lin], este lo conseguiremos con el la multiplicacion de kernel y del grad_output
        grad_outputs = grad_outputs.transpose(0,1) #[Cout,N,Lout]
        kernel = kernel.clone().view(Cout,-1).transpose(0,1) #[Cin*K,Cout]
        grad_inputs = kernel @ grad_outputs #[Cin*K,N,Lout], esto es lo que quiero para el fold: [N,Cin*K,Lout]
        grad_inputs = grad_inputs.transpose(0,1) #[N,Cin*K,Lout]
        grad_inputs = torch.nn.functional.fold(grad_inputs,output_size=(1,L_in),kernel_size=(1,K),stride=(1,1))

        return grad_inputs.squeeze(2), grad_weights
        

# ==== TEST ====
if __name__ == "__main__":
    torch.manual_seed(0)

    # 🔹 Inputs: N=2, Cin=2, Lin=6
    x = torch.tensor([
        [[1., 2., 3., 4., 5., 6.],     # Ejemplo 1, canal 1
         [6., 5., 4., 3., 2., 1.]],    # Ejemplo 1, canal 2

        [[0., 1., 2., 3., 4., 5.],     # Ejemplo 2, canal 1
         [5., 4., 3., 2., 1., 0.]]     # Ejemplo 2, canal 2
    ], requires_grad=True)  # shape (2, 2, 6)

    # 🔹 Kernel: Cout=3, Cin=2, kernel_size=3
    w = torch.tensor([
        [[1.0, 0.0, -1.0],     # Filtro 1, canal 1
         [0.0, 1.0, 0.0]],     # Filtro 1, canal 2

        [[0.5, 0.5, 0.5],      # Filtro 2, canal 1
         [-1.0, 0.0, 1.0]],    # Filtro 2, canal 2

        [[-1.0, 0.0, 1.0],     # Filtro 3, canal 1
         [1.0, 0.0, -1.0]]     # Filtro 3, canal 2
    ], requires_grad=True)  # shape (3, 2, 3)

    y = Conv1DFunctional.apply(x, w)
    print("Output shape:", y.shape)
    print("Output:", y)

    y.sum().backward()
    print("Grad input:", x.grad)
    print("Grad kernel:", w.grad)