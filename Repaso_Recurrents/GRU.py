import torch 
def sigmoid(x): 
    return (1)/(1+torch.exp(-x))

def tanh(x): 
    return (torch.exp(x) - torch.exp(-x))/(torch.exp(x) + torch.exp(-x))
class GRUTriDimFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs,h0:torch.Tensor,W_ih:torch.Tensor,W_hh:torch.Tensor,b_ih:torch.Tensor,b_hh:torch.Tensor): 
        """
            inputs is shape: [N,L,Hin]
            h0 is shape: [N,Hout]
            W_ih is shape: [3*Hout,Hin]
            W_hh is shape: [3*Hout,Hout]
            b_ih is shape: [3*Hout]
            b_hh is shape: [3*Hout]
            outputs shape is: [N,L,Hout]
        """
        N,L,Hin = inputs.shape 
        _,Hout = h0.shape 
        h_acumulacion = torch.empty(size=(N,L,Hout),dtype=inputs.dtype,device=inputs.device)
        for i in range(L): 
            h_prev = h0 if i == 0 else h_acumulacion[:,i-1,:]
            gi = inputs[:,i,:] @ W_ih.T + b_ih 
            gh = h_prev @ W_hh.T + b_hh 

            ri = gi[:,:Hout]
            rh = gh[:,:Hout]
            rt = sigmoid(ri + rh)

            zi = gi[:,Hout:2*Hout]
            zh = gh[:,Hout:2*Hout]
            zt = sigmoid(zi + zh)

            ni = gi[:,2*Hout:]
            nh = gh[:,2*Hout:]
            nt = tanh(ni + rt*nh)
            ht = (1 - zt)*nt + zt*h_prev 
            h_acumulacion[:,i,:] = ht 
        return h_acumulacion

