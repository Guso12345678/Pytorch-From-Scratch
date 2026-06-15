import torch 
def sigmoid(x):
    return 1/(1 + torch.exp(-x)) 
def tanh(x): 
    return (torch.exp(x) - torch.exp(-x))/(torch.exp(x) + torch.exp(-x))
class LSTMTriDimFunct(torch.autograd.Function): 

    @staticmethod
    def forward(ctx,inputs,h0:torch.Tensor,c0,weights_ih,bias_ih,weights_hh, bias_hh): 
        """
            inputs shape: [N,L,Hin]
            h_0 shape: [N,Hout]
            c_0 shape: [N, Hout]
            weights_ih shape: [4*Hout,Hin]
            weight_hh shape: [4*Hout,Hout]
            bias_ih shape: [4*Hout]
            bias_hh shape: [4*Hout]
        """
        N,L,Hin = inputs.shape 
        N,Hout = h0.shape 
        h_acumulacion = torch.empty(size=(N,L,Hout),device=h0.device,dtype=h0.dtype)
        c_acumulacion = torch.empty(size=(N,L,Hout),device=c0.device,dtype=c0.dtype)

        for i in range(L): 
            h_prev = h0 if i == 0 else h_acumulacion[:,i-1,:]
            c_prev = c0 if i == 0 else c_acumulacion[:,i-1,:]

            ni = inputs[:,i,:] @ weights_ih.T + bias_ih 
            nh = h_prev @ weights_hh.T + bias_hh 

            ii = ni[:,:Hout]
            ih = nh[:,:Hout]
            it = sigmoid(ii + ih)

            fi = ni[:,Hout:2*Hout]
            fh = nh[:,Hout:2*Hout]
            ft = sigmoid(fi + fh)

            gi = ni[:,2*Hout:3*Hout]
            gh = nh[:,2*Hout:3*Hout]
            gt = tanh(gi + gh)

            oi = ni[:,3*Hout:]
            oh = nh[:,3*Hout:]
            ot = sigmoid(oi + oh)

            ct = ft*c_prev + it*gt 
            ht = ot*tanh(ct) 

            h_acumulacion[:,i,:] = ht 
            c_acumulacion[:,i,:] = ct 

        return h_acumulacion[:,-1,:], c_acumulacion[:,-1,:]
    
            