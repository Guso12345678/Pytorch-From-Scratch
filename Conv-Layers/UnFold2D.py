import torch

class UnFold2DFunctional(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs:torch.Tensor,kernel_size:tuple,dilation:tuple,stride:tuple): 
        """
            Los inputs van a tener un shape de [N,C,H,W]
            la salida va a ser de shape [N, C*K, L_out]
        """
        N,C,H,W = inputs.shape 
        Kh, Kw = kernel_size 
        Dh, Dw = dilation
        Sh, Sw = stride 
        L_out_h = int(((H + 2*0 - Dh*(Kh - 1) -1 )/Sh) + 1)
        L_out_w = int(((W + 2*0 - Dw*(Kw - 1) - 1)/Sw) + 1) 
        L_out = L_out_h * L_out_w
        K = Kh * Kw 
        outputs = torch.empty((N,K*C,L_out))
        
        ctx.save_for_backward(torch.tensor(L_out)) 
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.stride = stride 

        for l in range(L_out): 
            i = l // L_out_w
            j = l % L_out_w
            h_start = i*Sh
            w_start = j*Sw
            for k in range(K): 
                kh = k // Kw
                kw = k % Kw
                h = h_start + kh*Dh
                w = w_start + kw*Dw

                outputs[:,k::K, l] = inputs[:,:,h,w]
    
    def backward(ctx,grad_outputs:torch.tensor): 
        """
            Ahora grad_outputs va a tener la forma de: [N,C*Kw*Kh,L_out_h*L_out_w*]
            nuestro inputs queremos que tenga la forma de: [N,C,H,W] 
        """
        L_out, = ctx.saved_tensors
        L_out = L_out.item() 
        kernel_size = ctx.kernel_size 
        dilation = ctx.dilation 
        stride = ctx.stride 
        Sh,Sw = stride 
        Kh,Kw = kernel_size
        Dh,Dw = dilation
        L_out_h,L_out_w = L_out

        K = Kh*Kw
        C = L_out // K 
        H = int(((L_out_h + 2*0 - Dh*(Kh - 1) - 1)/(Sh)) + 1)
        W = int(((L_out_w + 2*0 - Dw*(Kw - 1) - 1)/(Sw)) + 1)

        grad_inputs = torch.empty((grad_outputs.size(0),C,H,W))

        for l in range(L_out):
            i = l // L_out_w
            j = l % L_out_w
            h_start = i*Sh
            w_start = j*Sw 
            for k in range(K):
                kh =  k // Kw 
                kw = k % Kw 
                h = h_start + kh*Dh
                w = w_start + kw*Dw 

                grad_inputs[:,:,h,w] += grad_outputs[:,k::K,l]
        return grad_inputs, None, None, None  






