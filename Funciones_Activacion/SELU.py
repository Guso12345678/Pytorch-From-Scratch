import torch 

class SELUFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx:any,inputs:torch.Tensor,alpha:float,lamda:float): 
        ctx.save_for_backward(inputs)
        #PODRIAMOS PONER LOS ALPHA Y LAMBDA COMO UN TENSOR Y HACER LUEGO EN EL BACKWARD EL .ITEM() Y METERLO DENTRO DE LA FUNCION DEL CTX
        ctx.alpha = alpha
        ctx.lamda = lamda
        outputs = inputs.clone()
        outputs[inputs > 0] *= lamda
        outputs[inputs <= 0] = lamda*alpha*(torch.exp(inputs[inputs <=0 ])-1)
        return outputs

    @staticmethod
    def backward(ctx:any,grad_outputs:torch.Tensor): 
        inputs,  = ctx.saved_tensors
        alpha = ctx.alpha
        lamda = ctx.lamda
        grad_inputs = grad_outputs.clone()
        grad_inputs[inputs > 0] *= lamda
        grad_inputs[inputs <= 0] *= lamda*alpha*torch.exp(inputs[inputs <= 0])
        return grad_inputs, None, None

class SELU(torch.nn.Module): 
    def __init__(self,alpha:float=1.6733,lamda:float=1.0507): 
        super().__init__()
        self.fn = SELUFunction.apply
        self.alpha = alpha
        self.lamda = lamda
    
    def forward(self,inputs:torch.Tensor): 
        return self.fn(inputs,self.alpha,self.lamda)