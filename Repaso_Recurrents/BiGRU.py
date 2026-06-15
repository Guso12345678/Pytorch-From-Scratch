import torch 
def sigmoid(x): 
    return 1/(1 + torch.exp(-x))

def tanh(x): 
    return (torch.exp(x)- torch.exp(-x))/(torch.exp(x)+torch.exp(-x))
class BiGRUFunction(torch.autograd.Function): 

    @staticmethod
    def forward(ctx,inputs, h0_forward, h0_backward, W_ih_forward, W_hh_forward, b_ih_forward,b_hh_forward, W_ih_backward, W_hh_backward, b_ih_backward, b_hh_backward): 
        """
            shape inputs [N, L, Hin]
            shape h0_forward [N, Hout]
            shape h0_backward [N, Hout]
            shape W_ih_forward [3*Hout, Hin]
            shape W_hh_forward [3*Hout, Hout]
            shape b_ih_forward [3*Hout]
            shape b_hh_forward [3*Hout]
            shape W_ih_backward [3*Hout, Hin]
            shape W_hh_backward [3*Hout, Hout]
            shape b_ih_backward [3*Hout]
            shape b_hh_backward [3*Hout] 
        """ 
        N,L,Hin = inputs.shape 
        N,Hout = h0_forward.shape
        h_forward = torch.empty(size=(N,L,Hout))
        h_backward = torch.empty(size=(N,L,Hout))
        h_final = torch.empty(size=(N,2*Hout))

        for i in range(L): 
            h_prev = h0_forward if i == 0 else h_forward[:,i-1,:]
            gi = inputs[:,i,:] @ W_ih_forward.T + b_ih_forward 
            gh = h_prev @ W_hh_forward + b_hh_forward 

            ri = gi[:,:Hout]
            rh = gh[:,:Hout]
            rt = sigmoid(ri + rh)

            zi = gi[:,Hout:2*Hout]
            zh = gh[:,Hout:2*Hout]
            zt = sigmoid(zi + zh)

            ni = gi[:,2*Hout:]
            nh = gh[:,2*Hout:]
            nt = tanh(ni + rt*nh)

            ht = (1 -zt)*nt + zt*h_prev 
            h_forward[:,i,:] = ht 
        
        for i in reversed(range(L)): 
            h_back = h0_backward if i == (L - 1) else h_backward[:,i+1,:]
            
            gi = inputs[:,i,:] @ W_ih_backward.T + b_ih_backward 
            gh = h_back @ W_hh_backward + b_hh_backward 

            ri = gi[:,:Hout]
            rh = gh[:,:Hout]
            rt = sigmoid(ri + rh)

            zi = gi[:,Hout:2*Hout]
            zh = gh[:,Hout:2*Hout]
            zt = sigmoid(zi + zh)

            ni = gi[:,2*Hout:]
            nh = gh[:,2*Hout:]
            nt = tanh(ni + rt*nh)

            ht = (1 -zt)*nt + zt*h_back 
            h_backward[:,i,:] = ht 
        
        h_f = h_forward[:,-1,:]
        h_b = h_backward[:,0,:]
        h_final[:,:Hout] = h_f 
        h_final[:,Hout:] = h_b 
        return h_final 
