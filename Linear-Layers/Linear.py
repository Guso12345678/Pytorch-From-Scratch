import torch 

class LinearFunction(torch.autograd.Function): 
    
    @staticmethod
    def forward(ctx,inputs,weights,bias): 
        """
            the inputs shape is: [N,C,Hin]
            the weight shape is: [Hout,Hin]
            the outputs shape is: [N,C,Hout]
        """
        output = inputs @ weights.T + bias
        ctx.save_for_backward(inputs,weights) 
        return output 
    
    @staticmethod
    def backward(ctx,grad_outputs:torch.Tensor):
        """
            The grad_inputs shape is: [N,C,Hin]
            The grad_weights shape is: [Hout,Hin]
            The grad_bias shape is: [Hout]
            The shape of grad_outputs is: [N,C,Hout]
        """ 
        inputs,weights = ctx.save_tensors 
        N,C,Hin = inputs.shape
        _,_,Hout = grad_outputs.shape 
        #Primero Sacamos el grad_inputs: 
        grad_inputs = grad_outputs @ weights #Salida de [N,C,Hin]

        #Ahora Sacamos el grad_weights: 
        grad_weights = grad_outputs.permute(2,1,0).reshape(Hout,N*C) @ inputs.reshape(N*C,Hin)

        #Por ultimo Sacamos el grad_bias: 
        grad_bias = torch.sum(grad_outputs.reshape(N*C,Hout),dim=0)

        return grad_inputs, grad_weights, grad_bias  



