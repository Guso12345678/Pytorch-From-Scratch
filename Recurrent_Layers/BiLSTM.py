import torch 
def sigmoid(x): 
    return 1/(1 + torch.exp(-x))

def tanh(x): 
    return (torch.exp(x)- torch.exp(-x))/(torch.exp(x)+torch.exp(-x))
class BiLSTMFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs, h0_forward,c0_forward, h0_backward,c0_backward, W_ih_forward, W_hh_forward, b_ih_forward,b_hh_forward, W_ih_backward, W_hh_backward, b_ih_backward, b_hh_backward): 
        """
            shape inputs [N, L, Hin]
            shape h0_forward [N, Hout]
            shape h0_backward [N, Hout]
            shape W_ih_forward [4*Hout, Hin]
            shape W_hh_forward [4*Hout, Hout]
            shape b_ih_forward [4*Hout]
            shape b_hh_forward [4*Hout]
            shape W_ih_backward [4*Hout, Hin]
            shape W_hh_backward [4*Hout, Hout]
            shape b_ih_backward [4*Hout]
            shape b_hh_backward [4*Hout] 
        """
        N,L,_ = inputs.shape 
        _,Hout = h0_forward.shape
        h_acumulacion_forward = torch.empty(size=(N,L,Hout),dtype=h0_forward.dtype,device=h0_forward.device)
        h_acumulacion_backward = torch.empty(size=(N,L,Hout), dtype=h0_backward.dtype, device=h0_backward.device)
        c_acumulacion_forward = torch.empty(size=(N,L,Hout),dtype=c0_forward.dtype, device=c0_forward.device)
        c_acumulacion_backward = torch.empty(size=(N,L,Hout),dtype=c0_backward.dtype, device=c0_backward.device)
        h_final = torch.empty((N,2*Hout)) 
        c_final = torch.empty((N,2*Hout))

        ###Primero sacamos los relacionado con el forward: 
        for i in range(L): 
            h_prev = h0_forward if i==0 else h_acumulacion_forward[:,i-1,:]
            c_prev = c0_forward if i==0 else c_acumulacion_forward[:,i-1,:]
            ni = inputs[:,i,:] @ W_ih_forward.T + b_ih_forward
            nh = h_prev @ W_hh_forward.T + b_hh_forward

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

            h_acumulacion_forward[:,i,:] = ht 
            c_acumulacion_forward[:,i,:] = ct 

        ###AHORA HACEMOS TOD LO RELACIONADO CON EL BACKWARD: 
        for i in reversed(range(L)): 
            h_prev = h0_backward if i == (L - 1) else h_acumulacion_backward[:,i+1,:]
            c_prev = c0_backward if i == (L - 1) else c_acumulacion_backward[:,i+1,:]

            ni = inputs[:,i,:] @ W_ih_backward.T + b_ih_backward
            nh = h_prev @ W_hh_backward.T + b_hh_backward

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

            h_acumulacion_backward[:,i,:] = ht 
            c_acumulacion_backward[:,i,:] = ct
        
        hn_forward = h_acumulacion_forward[:,-1,:]
        hn_backward = h_acumulacion_backward[:,0,:]
        cn_forward = c_acumulacion_forward[:,-1,:]
        cn_backward = c_acumulacion_backward[:,0,:]
        h_final[:,:Hout] = hn_forward
        h_final[:,Hout:] = hn_backward
        c_final[:,:Hout] = cn_forward
        c_final[:,Hout:] = cn_backward

        return h_final, c_final 


# Parámetros de ejemplo
N = 4           # batch size
L = 5           # secuencia de entrada
Hin = 10        # dimensión de entrada
Hout = 8        # dimensión del estado oculto

# Entradas simuladas
inputs = torch.randn(N, L, Hin, requires_grad=True)
h0_f = torch.randn(N, Hout, requires_grad=True)
c0_f = torch.randn(N, Hout, requires_grad=True)
h0_b = torch.randn(N, Hout, requires_grad=True)
c0_b = torch.randn(N, Hout, requires_grad=True)

# Pesos y bias forward
W_ih_f = torch.randn(4 * Hout, Hin, requires_grad=True)
W_hh_f = torch.randn(4 * Hout, Hout, requires_grad=True)
b_ih_f = torch.randn(4 * Hout, requires_grad=True)
b_hh_f = torch.randn(4 * Hout, requires_grad=True)

# Pesos y bias backward
W_ih_b = torch.randn(4 * Hout, Hin, requires_grad=True)
W_hh_b = torch.randn(4 * Hout, Hout, requires_grad=True)
b_ih_b = torch.randn(4 * Hout, requires_grad=True)
b_hh_b = torch.randn(4 * Hout, requires_grad=True)

# Ejecutar forward
h_final, c_final = BiLSTMFunction.apply(
    inputs, h0_f, c0_f, h0_b, c0_b,
    W_ih_f, W_hh_f, b_ih_f, b_hh_f,
    W_ih_b, W_hh_b, b_ih_b, b_hh_b
)

# Solo para ver resultados
print("Output h_final:", h_final.shape)
print("Output c_final:", c_final.shape)




            

