import torch 

def tanh(x): 
    return (torch.exp(x) - torch.exp(-x))/(torch.exp(x) + torch.exp(-x))

class RNNCellFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs,hx,weight_i,bias_i,weights_h,bias_h): 
        """
            the inputs shape is: [batch_size, input_size]
            the h{t-1} shape is: [batch_size, hidden_size]
            the weight_i shape is: [hidden_size,input_size]
            the bias_i shape is: [hidden_size]
            the weights_h shape is: [hidden_size,hidden_size]
            the bias_h shape is: [hidden_size]
        """
        gi = inputs @ weight_i.T + bias_i
        gh = hx @ weights_h.T + bias_h
        g = gi + gh  
        h = tanh(g)
        ctx.save_for_backward(inputs,h,hx,g,weight_i,weights_h)
        return h

    def backward(ctx,grad_outputs): 
        """
            grad_outputs is shape: [batch_size,hidden_size]
            h is shape [batch_size, hidden_size]
            h0 is shape [batch_size, hidden_size]
        """
        #Como en cada instante de tiempo y en este caso como es una unica celula, lo que se va a utilizar para la actualizacion de los todos los pesos no es solo el grad_outputs 
        #Sino la suma de grad_outputs y h
        inputs, h, h0, g , weight_i , weights_h = ctx.saved_tensors 
        #grad_final = grad_outputs + h #Suma de las dos cosas que le entran al gradiente

        #Sacamos el gradiente de la tanh que sera necesaria para todo:
        grad_tanh = (1 - tanh(g)**2)*grad_outputs #Shape de [batch_size,hidden_size]

        #gradiente de weights_i 
        grad_weight_i = grad_tanh.T @ inputs

        #gradiente de inputs 
        grad_inputs = grad_tanh @ weight_i

        #gradiente de bias_i 
        grad_bias_i = torch.sum(grad_tanh,dim=0)

        #gradiente de weights_h 
        grad_weight_h = grad_tanh.T @ h0 

        #gradiente de bias_h 
        grad_bias_h = torch.sum(grad_tanh, dim=0)

        #gradiente de h 
        grad_h = grad_tanh @ weights_h

        return grad_inputs, grad_h, grad_weight_i, grad_bias_i, grad_weight_h, grad_bias_h






        


        
          
