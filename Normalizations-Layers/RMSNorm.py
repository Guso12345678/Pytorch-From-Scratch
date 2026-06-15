import torch

class RMSNormFunction(torch.autograd.Function): 

    @staticmethod
    def forward(ctx,inputs:torch.Tensor,eps:float,gamma:torch.Tensor): 
        """
            inputs of shape: [N,C,Hin,Win]
            gamma of shape: [C]
        """
        #Primero sacamos el termino de RMS que sera por lo que se dividir
        rms_term = torch.sqrt(torch.mean(torch.pow(inputs,2),dim=(2,3),keepdim=True) + eps)

        #Ahora aplicamso la normalizacion:
        inputs_hat = inputs / rms_term

        #Por ultimo ya: aplicamos el gamma que vamos a expandirlo para poder aplicar broadcasting: 
        gamma_expanded = gamma.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        #LA salida es la multiplicacion de ambos: 
        outputs = inputs_hat * gamma_expanded
        ctx.save_for_backward(inputs,inputs_hat, rms_term, gamma)
        ctx.eps = eps

        return outputs 
    
    @staticmethod
    def backward(ctx,grad_outputs:torch.Tensor): 
        """
            the grad_outputs is shape: [N,C,Hin,Win]
            the grad_inputs is shape: [N,C,Hin,Win]
            the grad_gamma is shape: [C]
        """
        inputs,inputs_hat, rms_term, gamma = ctx.saved_tensors

        #Comenzamos primero con la derivada de grad_inputs_hat, que es como la derivada intermedia:
        gamma_expanded = gamma.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        grad_inputs_hat = grad_outputs * gamma_expanded

        grad_inputs = (1/rms_term)*(grad_inputs_hat - ((inputs*torch.mean(inputs*grad_inputs_hat,dim=(2,3),keepdim=True))/(torch.pow(rms_term,2))))

        #Ahora sacamos el gradiente de la gamma: 
        grad_gamma = torch.sum(inputs_hat*grad_outputs,dim=(0,2,3))

        return grad_inputs, None, grad_gamma



