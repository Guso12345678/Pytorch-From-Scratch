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
        triple_hout,hin = W_ih.shape 
        H_out = triple_hout // 3 
        N,L,Hin = inputs.shape 

        outputs = torch.zeros(size=(N,L,H_out),device=inputs.device,dtype=inputs.dtype)

        for i in range(L): 
            h_prev = outputs[:,i-1,:] if i > 0 else h0 
            
            g_i = inputs[:,i,:] @ W_ih.T + b_ih #The output shape is: [N,3*Hout]
            g_h = h_prev @ W_hh.T + b_hh #The output shape is: [N,3*Hout]

            i_r = g_i[:,:H_out]
            i_z = g_i[:,H_out:2*H_out]
            i_n = g_i[:,2*H_out:]

            h_r = g_h[:,:H_out]
            h_z = g_h[:,H_out:2*H_out]
            h_n = g_h[:,2*H_out:]

            rt = sigmoid(i_r + h_r)
            zt = sigmoid(i_z + h_z)
            nt = tanh(i_n + rt*(h_n))

            outputs[:,i,:] = (1 - zt)*nt + zt*h_prev
        return outputs 


def main():
    torch.manual_seed(0)

    N, L, Hin, Hout = 2, 4, 3, 5  # batch, seq, in, hidden

    x = torch.randn(N, L, Hin)         # [N, L, Hin]
    h0 = torch.randn(N, Hout)          # [N, Hout]

    W_ih = torch.randn(3 * Hout, Hin) * 0.1
    W_hh = torch.randn(3 * Hout, Hout) * 0.1
    b_ih = torch.randn(3 * Hout) * 0.1
    b_hh = torch.randn(3 * Hout) * 0.1

    y = GRUTriDimFunction.apply(x, h0, W_ih, W_hh, b_ih, b_hh)

    print("Output [N, L, Hout]:")
    print(y)

if __name__ == "__main__": 
    main()
