import torch

class MaxPool2dNonEfficient(torch.autograd.Function):
    @staticmethod
    def forward(ctx,inputs:torch.Tensor,padding,kernel_size,stride,dilation): 
        """
            the shape of the inputs is: [N,C,H,W]
            the shape of the desired outputs is: [N,C,Hout,Wout]
        """
        N,C,H,W = inputs.shape
        Kh,Kw = kernel_size
        Ph,Pw = padding
        Sh,Sw = stride 
        Dh,Dw = dilation 
        if Ph > 0 or Pw > 0:
            inputs = torch.nn.functional.pad(inputs, (Pw, Pw, Ph, Ph))

        H_padded, W_padded = inputs.shape[2], inputs.shape[3]
        Hout = int(((H_padded - Dh * (Kh - 1) - 1) / Sh) + 1)
        Wout = int(((W_padded - Dw * (Kw - 1) - 1) / Sw) + 1)
        outputs = torch.empty((N,C,Hout,Wout))
        
        for i in range(Hout): 
            for j in range(Wout): 
                h_start = i*Sh
                w_start = j*Sw
                h_end = h_start + Kh*Dh
                w_end = w_start + Kw*Dw
                window = inputs[:, :, h_start:h_end,w_start:w_end] # (N, C, Kh, Kw)
                window_shaped = window.reshape(N,C,-1)
                outputs[:,:,i,j] = torch.max(window_shaped,dim=-1).values 



