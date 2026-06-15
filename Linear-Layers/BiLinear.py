import torch

class BiLinearFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, inputs1, inputs2, weights, bias): 
        """
        inputs1: [N, C, Hin1]
        inputs2: [N, C, Hin2]
        weights: [Hout, Hin1, Hin2]
        bias:    [Hout]
        Returns: [N, C, Hout]
        """
        N, C, Hin1 = inputs1.shape
        _, _, Hin2 = inputs2.shape
        Hout = weights.shape[0]

        # Paso 1: Reorganizar entradas a [N*C, Hin1/Hin2]
        x1 = inputs1.reshape(N * C, Hin1)
        x2 = inputs2.reshape(N * C, Hin2)

        # Paso 2: Expandimos x1 y x2 para operar con los pesos
        # x1 : [N*C, 1, Hin1], weights → [Hout, Hin1, Hin2]
        x1_exp = x1.unsqueeze(1)  # [N*C, 1, Hin1]
        x2_exp = x2.unsqueeze(1)  # [N*C, 1, Hin2]

        # Paso 3: expandir pesos para multiplicar por lotes
        # w : [1, Hout, Hin1, Hin2]
        W = weights.unsqueeze(0)  # [1, Hout, Hin1, Hin2]

        # Paso 4: hacer multiplicación bilineal: x1 @ W @ x2^T
        # x1 @ W :  [N*C, Hout, Hin2]
        x1W = torch.matmul(x1_exp, W)  # [N*C, Hout, 1, Hin2]

        # (x1W * x2) : [N*C, Hout, 1] sumando sobre Hin2
        x1Wx2 = (x1W.squeeze(2) * x2_exp).sum(dim=-1)  # [N*C, Hout]

        # Paso 5: añadir bias y devolver
        output = x1Wx2 + bias  #broadcasting
        output = output.view(N, C, Hout)

        ctx.save_for_backward(inputs1, inputs2, weights)
        ctx.bias = bias
        return output 
    @staticmethod
    def backward(ctx,grad_outputs:torch.Tensor): 
        """
            shape of grad_outputs : [N,C,Hout]
            shape of grad_inputs1 : [N,C,Hin1]
            shape of grad_inputs2 : [N,C,Hin2]
            shape of grad_weights : [Hout,Hin1,Hin2]
            shape of grad_bias : [Hout]
        """
        inputs1,inputs2,weights = ctx.saved_tensors 
        N,C,Hout = grad_outputs.shape 
        N,C,Hin1 = inputs1.shape 
        N,C,Hin2 = inputs2.shape 
        Hout,Hin1,Hin2 = weights.shape 
        x1 = inputs1.reshape(N*C, Hin1)    # [B, Hin1]
        x2 = inputs2.reshape(N*C, Hin2)    # [B, Hin2]
        grad_outputs = grad_outputs.reshape(N*C, Hout)  # [B, Hout]


        ###PRIMERA FORMA MEDIANTE EINSUM: 
        # grad_inputs1 = torch.einsum('bo,oij,bj -> bi',grad_outputs,weights,x2)
        # grad_inputs2 = torch.einsum('bo,oij,bi -> bj',grad_outputs,weights,x1)
        # grad_weights = torch.einsum('bo,bi,bj -> oij',grad_outputs,x1,x2)

        # grad_bias = torch.sum(grad_outputs,dim=0)


        #SEGUNDA FORMA NORMAL SIN EL USO DE EINSUM:
        # Primero sacamos el grad_inputs1:  
        W_flat = weights.reshape(Hout, Hin1 * Hin2)
        grad_w = grad_outputs @ W_flat
        grad_w = grad_w.reshape(-1, Hin1, Hin2)
        grad_inputs1 = (grad_w * x2.unsqueeze(1)).sum(dim=2)
        grad_inputs1 = grad_inputs1.view(N, C, Hin1)

        #Segundo sacamos el grad_inputs2: 
        grad_inputs2 = (grad_w * x1.unsqueeze(1)).sum(dim=-1)
        grad_inputs2 = grad_inputs2.view(N,C,Hin2)

        #Tercer paso sacamos el grad_weights:
        grad_weights = torch.zeros_like(weights)
        for o in range(weights.shape[0]):
            go = grad_outputs[:, o].unsqueeze(1)
            grad_weights[o] = (go * x1).T @ x2

        #Cuarto paso el grad_bias: 
        grad_bias = torch.sum(grad_outputs,dim=0)



        return grad_inputs1, grad_inputs2, grad_weights, grad_bias
    
    
        




