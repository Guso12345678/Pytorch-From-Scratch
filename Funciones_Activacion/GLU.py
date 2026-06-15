import torch 

def sigmoid(tensor): 
    return (1)/(1+torch.exp(-tensor))
class GLUFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs): 
        _,N,_ = inputs.shape 
        division = N // 2
        a = inputs[:, :division, :]
        b = inputs[:, division:, :]
        ctx.save_for_backward(inputs,a,b)
        outputs = a*sigmoid(b)
        return outputs 
    
    @staticmethod
    def backward(ctx,grad_outputs): 
        """
            Como los outputs tendran de shape N/2 al ser la multiplicacion de a y b un producto elemento a elemento.
        """
        inputs,a,b = ctx.saved_tensors 
        grad_a = sigmoid(b)*grad_outputs
        grad_b = a*(sigmoid(b)*(1-sigmoid(b)))
        _,split,_ = grad_a.shape
        """ASI ES COMO SE HACE MANUALMENTE EN CONCAT DE DOS TENSORES"""
        # grad_inputs = torch.zeros_like(inputs)
        # grad_inputs[:,:split,:] = grad_a
        # grad_inputs[:,split:,:] = grad_b
        grad_inputs = torch.concat([grad_a,grad_b],dim=1)
        return grad_inputs,None,None
