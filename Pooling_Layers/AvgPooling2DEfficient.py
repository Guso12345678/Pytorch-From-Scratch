import torch
class AvgPooling2DEfficient(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs:torch.Tensor,kernel_size,stride,padding): 
        """
            The shape of inputs is:[N,C,H,W]
        """
        N,C,H,W = inputs.shape 
        inputs_unfolded = inputs.unfold(dimension=2,size=kernel_size,step=stride).unfold(dimension=3,size=kernel_size,step=stride)#Shape: [N,C,Hout,Wout,Kw,Kh]
        N,C,Hout,Wout,Kh,Kw = inputs_unfolded.shape 
        inputs_unfolded = inputs_unfolded.view(N,C,Hout,Wout,-1)
        mean = inputs_unfolded.mean(dim=-1) #[N,C,Hout,Wout] 
        ctx.kernel_size = kernel_size
        ctx.H = H 
        ctx.W = W 
        ctx.stride = stride
        ctx.padding = padding
        ctx.save_for_backward(inputs)
 
        return mean 
    @staticmethod
    def backward(ctx,grad_outputs:torch.Tensor): 
        """
            the shape of grad_outputs is: [N,C,Hout,Wout]
        """
        kernel_size = ctx.kernel_size 
        stride = ctx.stride 
        padding = ctx.padding 
        inputs, = ctx.saved_tensors
        W = ctx.W 
        H = ctx.H  
        N,C,Hout,Wout = grad_outputs.shape

        Kh,Kw = kernel_size
        escalado = 1/(Kh*Kw)
        
        grad_inputs = escalado*grad_outputs #Con esto tenemos un shape de:[N,C,Hout,Wout] 
        grad_inputs = grad_inputs.unsqueeze(-1).expand(-1, -1, -1, -1, Kh * Kw)  #Primero con el unsqueeze se añade una dimension al final: [N,C,Hout,Wout,1] y luego con el expand replicas el valor del gradiente uno por cada ventana que has tenido; obteniendo: [N,C,Hout,Wout,K]  
        grad_inputs = grad_inputs.permute(0,1,4,2,3).reshape(N, C * Kh * Kw, Hout * Wout)#Ahora se reordena para que te quede: [N,C,K,Hout,Wout] y tendras que agrupar para que este en el formato correcto con el que puede trabajar el fold: [N,C*K,Hout*Wout] donde la K= Kw*Kh
        grad_inputs_folded = torch.nn.functional.fold(grad_inputs,output_size=(H,W),kernel_size=(Kh,Kw),padding=padding,stride=stride) #Se aplica el fold que se encarga de plegarlo todo, obteniendo la dimension deseada: [N,C,H,W]
        return grad_inputs_folded, None, None, None, None, None   
