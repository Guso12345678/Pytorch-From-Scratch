import torch 

class AvgPooling2DNonEfficient(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs,kernel_size,padding,stride,dilation): 
        """ 
            inputs shape: [N,C,H,W] 
            and outputs_shape: [N,C,Hout,Wout]
        """
        N,C,H,W = inputs.shape 
        Kh,Kw = kernel_size
        Ph,Pw = padding 
        Sh,Sw = stride 
        Dh,Dw = dilation
        Hout = int(((H + 2*Ph - Kh)/(Sh)) + 1)
        Wout = int(((W + 2*Pw - Kw)/(Sw)) + 1)

        outputs = torch.empty((N,C,Hout,Wout))

        for i in range(Hout): 
            for j in range(Wout): 
                hstart = i*Sh 
                wstart = j*Sw 
                h = hstart + Kh*Dh 
                w = wstart + Kw*Dw
                window = inputs[:,:,hstart:h,wstart:w]
                window_reshaped:torch.Tensor = window.view(0,1,-1)
                outputs[:,:,i,j] = window_reshaped.mean(dim=-1)
        return outputs 