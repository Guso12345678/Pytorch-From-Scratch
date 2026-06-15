import torch 
def tanh(x): 
    return (torch.exp(x) - torch.exp(-x))/(torch.exp(x) + torch.exp(-x))
class BiRNNFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs,h0_forward,h0_backward,W_ih_forward,W_hh_forward,b_ih_forward,b_hh_forward,W_ih_backward,W_hh_backward,b_ih_backward,b_hh_backward): 
        """
            the inputs shape is: [batch_size,L ,input_size]
            the h0_forward shape is: [batch_size, hidden_size]
            the h0_backward shape is: [batch_size, hidden_size]
            the W_ih_forward shape is: [hidden_size,input_size]
            the W_hh_forward shape is: [hidden_size,hidden_size]
            the b_ih_forward shape is: [hidden_size]
            the b_hh_forward shape is: [hidden_size]
            the W_ih_backward shape is: [hidden_size,input_size]
            the W_hh_backward shape is: [hidden_size,hidden_size]
            the b_ih_backward shape is: [hidden_size]
            the b_hh_backward shape is: [hidden_size]
        """
        B,L,_ = inputs.shape 
        _,Hout = h0_forward.shape 
        h_acumulacion_forward = torch.empty(size=(B,L,Hout),dtype=h0_forward.dtype,device=h0_forward.device)
        h_acumulacion_backward = torch.empty(size=(B,L,Hout),dtype=h0_backward.dtype,device=h0_backward.device)
        h_final = torch.empty(size=(B,2*Hout))
        for i in range(L): 
            h_prev = h0_forward if i==0 else h_acumulacion_forward[:,i-1,:]
            
            gi = inputs[:,i,:] @ W_ih_forward.T + b_ih_forward 
            gh = h_prev @ W_hh_forward.T + b_hh_forward 

            ht = tanh(gi + gh)

            h_acumulacion_forward[:,i,:] = ht 
        
        for i in reversed(range(L)): 
            h_prev = h0_backward if i == (L - 1) else h_acumulacion_backward[:,i+1,:]

            gi = inputs[:,i,:] @ W_ih_backward.T + b_ih_backward 
            gh = h_prev @ W_hh_backward.T + b_hh_backward 

            ht = tanh(gi + gh)

            h_acumulacion_backward[:,i,:] = ht

        hn_forward = h_acumulacion_forward[:,-1,:]
        hn_backward = h_acumulacion_backward[:,0,:]
        h_final[:,:Hout] = hn_forward
        h_final[:,Hout:] = hn_backward

        return h_final 

batch_size = 4
seq_len = 6
input_size = 10
hidden_size = 8

# Entradas aleatorias
inputs = torch.randn(batch_size, seq_len, input_size, requires_grad=True)
h0_forward = torch.randn(batch_size, hidden_size, requires_grad=True)
h0_backward = torch.randn(batch_size, hidden_size, requires_grad=True)

W_ih_forward = torch.randn(hidden_size, input_size, requires_grad=True)
W_hh_forward = torch.randn(hidden_size, hidden_size, requires_grad=True)
b_ih_forward = torch.randn(hidden_size, requires_grad=True)
b_hh_forward = torch.randn(hidden_size, requires_grad=True)

W_ih_backward = torch.randn(hidden_size, input_size, requires_grad=True)
W_hh_backward = torch.randn(hidden_size, hidden_size, requires_grad=True)
b_ih_backward = torch.randn(hidden_size, requires_grad=True)
b_hh_backward = torch.randn(hidden_size, requires_grad=True)

# Ejecutamos forward
output = BiRNNFunction.apply(
    inputs, h0_forward, h0_backward,
    W_ih_forward, W_hh_forward, b_ih_forward, b_hh_forward,
    W_ih_backward, W_hh_backward, b_ih_backward, b_hh_backward
)

# Mostramos resultado y probamos backward
print("Output shape:", output.shape)
print("Output: ",output)  # Esperado: [batch_size, 2*hidden_size]

