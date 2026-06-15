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
        N,L,_ = inputs.shape 
        _,Hout = h0_forward.shape
        h_acumulacion_forward = torch.empty(size=(N,L,Hout),dtype=h0_forward.dtype,device=h0_forward.device)
        h_acumulacion_backward = torch.empty(size=(N,L,Hout), dtype=h0_backward.dtype, device=h0_backward.device)
        h_final = torch.empty((N,2*Hout))
        ###PRIMERO SACAMOS TOD RELACIONADO CON EL FORWARD: 
        for i in range(L): 
            h_prev = h0_forward if i == 0 else h_acumulacion_forward[:,i-1,:]
            fi = inputs[:,i,:] @ W_ih_forward.T + b_ih_forward 
            fh = h_prev @ W_hh_forward.T + b_hh_forward 

            ri = fi[:,:Hout]
            rh = fh[:,:Hout]
            rt = sigmoid(ri + rh)

            zi = fi[:,Hout:2*Hout] 
            zh = fh[:,Hout:2*Hout]
            zt = sigmoid(zi + zh)

            ni = fi[:,2*Hout:]
            nh = fh[:,2*Hout:]
            nt = tanh(ni + rt*nh)

            h_acumulacion_forward[:,i,:] = (1 - zt)*nt + zt*h_prev

        ###AHORA SACAMOS TOD RELACIONADO CON EL BACKWARD 
        for i in reversed(range(L)): 
            h_prev = h0_backward if i == (L - 1) else h_acumulacion_backward[:,i+1,:]
            fi = inputs[:,i,:] @ W_ih_backward.T + b_ih_backward 
            fh = h_prev @ W_hh_backward.T + b_hh_backward 

            ri = fi[:,:Hout]
            rh = fh[:,:Hout]
            rt = sigmoid(ri + rh)

            zi = fi[:,Hout:2*Hout] 
            zh = fh[:,Hout:2*Hout]
            zt = sigmoid(zi + zh)

            ni = fi[:,2*Hout:]
            nh = fh[:,2*Hout:]
            nt = tanh(ni + rt*nh)

            h_acumulacion_backward[:,i,:] = (1 - zt)*nt + zt*h_prev
        
        h_t_forward = h_acumulacion_forward[:,-1,:]
        h_t_backward = h_acumulacion_backward[:,0,:]
        h_final[:,:Hout] = h_t_forward
        h_final[:,Hout:] = h_t_backward
        return h_final  


def main():
    torch.manual_seed(0)

    # Tamaños
    N, L, Hin, Hout = 2, 5, 10, 6

    # Entradas
    inputs = torch.randn(N, L, Hin)
    h0_f = torch.randn(N, Hout)
    h0_b = torch.randn(N, Hout)

    # Parámetros forward
    W_ih_f = torch.randn(3 * Hout, Hin)
    W_hh_f = torch.randn(3 * Hout, Hout)
    b_ih_f = torch.randn(3 * Hout)
    b_hh_f = torch.randn(3 * Hout)

    # Parámetros backward
    W_ih_b = torch.randn(3 * Hout, Hin)
    W_hh_b = torch.randn(3 * Hout, Hout)
    b_ih_b = torch.randn(3 * Hout)
    b_hh_b = torch.randn(3 * Hout)

    # Llama a tu BiGRUFunction
    output = BiGRUFunction.apply(
        inputs, h0_f, h0_b,
        W_ih_f, W_hh_f, b_ih_f, b_hh_f,
        W_ih_b, W_hh_b, b_ih_b, b_hh_b
    )

    print("Output shape:", output.shape)  # Esperado: [N, L, 2*Hout]

main()





