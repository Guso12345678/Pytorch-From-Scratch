import torch 

class RestNetConv1DFunction(torch.autograd.Function): 

    @staticmethod
    def forward(ctx,inputs:torch.Tensor,weights:torch.Tensor,bias:torch.Tensor,padding=1,dilation=1,stride=1): 
        """
            The inputs shape is: [N,C_in,L_in]
            the weights shape is: [C_out,C_in,K]
            the bias shape is: [C_out]
            the outputs shape is: [N,C_out,L_out]
        """
        N,C_in,L_in = inputs.shape 
        C_out,C_in,K = weights.shape 
        L_out = int(((L_in + 2*padding - dilation*(K - 1) - 1)/stride) + 1)

        inputs_padded = torch.nn.functional.pad(inputs,(1,1))

        inputs_unfolded = torch.nn.functional.unfold(inputs_padded.unsqueeze(2),kernel_size=(1,K),dilation=(1,dilation),padding=(0,0),stride=(1,stride)) #shape: [N,C_in*K,L_out]
        output = inputs_unfolded.transpose(1,2) @ weights.clone().reshape(C_out,C_in*K).T + bias #Output shape [N,L_out,C_out]
        output_final = output.transpose(1,2)
        ctx.save_for_backward(inputs_unfolded,weights,bias)
        ctx.padding = padding 
        ctx.dilation = dilation 
        ctx.C_in = C_in 
        ctx.K = K 
        ctx.stride = stride 
        return output_final + inputs
    
    @staticmethod
    def backward(ctx,grad_outputs:torch.Tensor): 
        """
            the shape of grad_outputs is: [N,C_out,L_out]
            the shape of grad_weights is: [C_out,C_in,K]
            the shape of grad_inputs is: [N,C_in,L_in]
            the shape of grad_bias is: [C_out]
        """
        inputs_unfolded, weights, bias = ctx.saved_tensors #Inputs_unfolded shape is: [N,C_in*K,L_out] 
        padding = ctx.padding 
        stride = ctx.stride
        dilation = ctx.dilation 
        C_in = ctx.C_in 
        K = ctx.K 
        N,CT,Lout = inputs_unfolded.shape 
        N,Cout,_ = grad_outputs.shape 

        ###PRIMERO VAMOS A CALCULAR EL GRADIENTE DE LOS PESOS: 
        #Primero preparamos los inputs para la multiplicacion: 
        inputs_unfolded_reshaped = inputs_unfolded.transpose(1,2).reshape(N*Lout,CT).T #Obtenemos: [C_in*K,N*L_out]
        grad_outputs_reshaped = grad_outputs.transpose(1,2).reshape(N*Lout,Cout)
        
        grad_weights = inputs_unfolded_reshaped @ grad_outputs_reshaped #Obtenemos: [C_in*K,Cout] 
        grad_weights_salida = grad_weights.transpose(0,1).reshape(Cout,C_in,K)

        ###AHORA VAMOS A CALCULAR EL GRADIENTE DE LOS INPUTS: 
        #La entrada que va a recibir el fold tendra que ser del shape de: [N,C_in*K,L_out] 
        grad_outputs_prep = grad_outputs.transpose(1,2).reshape(N*Lout,C_out) 
        weights_prep = weights.reshape(C_out,C_in*K)
        grad_inputs_before_fold = grad_outputs_prep @ weights_prep #the shape is: [N*Lout,C_in*K] 
        grad_inputs_before_fold = grad_inputs_before_fold.reshape(N,Lout,C_in*K).transpose(1,2) #the shape is perfect for the fold: [N,C_in*K,L_out]

        grad_inputs = torch.nn.functional.fold(grad_inputs_before_fold,output_size=(1, L_in),kernel_size=(1, K),padding=(0, padding),stride=(1, stride),dilation=(1, dilation)) #the shape is: [N,C_in,1,L_in(despues del padding)]
        grad_inputs_final = grad_inputs.squeeze(2) + grad_outputs #Se añade el gradiente de los outputs para asi poder añadir el termino del la conexion residual. 

        ###Por ultimo el grad_bias que es unicamente la suma en las dos dimensiones tanto de N,L_out 
        grad_outputs_prep_for_bias = grad_outputs.transpose(1,2)
        grad_bias = grad_outputs_prep_for_bias.sum(dim=(0,1))

        return grad_inputs_final, grad_weights_salida, grad_bias, None, None, None 




        



        




# Main para debuggear
if __name__ == "__main__":
    torch.manual_seed(0)

    N, C_in, L_in = 1, 2, 5
    C_out, K = 2, 3

    x = torch.randn(N, C_in, L_in, requires_grad=True)
    w = torch.randn(C_out, C_in, K, requires_grad=True)
    b = torch.randn(C_out, requires_grad=True)

    # Apply custom function
    y = RestNetConv1DFunction.apply(x, w, b)
    print("Output:\n", y)

    # Backward
    loss = y.sum()
    loss.backward()

    print("\nGrad Input:\n", x.grad)
    print("Grad Weights:\n", w.grad)
    print("Grad Bias:\n", b.grad)