import torch 
def sigmoid(x): 
    return (1)/(1+torch.exp(-x))

def tanh(x): 
    return (torch.exp(x) - torch.exp(-x))/(torch.exp(x) + torch.exp(-x))

class GRUFunction(torch.autograd.Function): 
    
    @staticmethod
    def forward(ctx,inputs:torch.Tensor,h0:torch.Tensor,W_ir:torch.Tensor,b_ir:torch.Tensor,W_hr:torch.Tensor,b_hr:torch.Tensor,W_iz:torch.Tensor,b_iz:torch.Tensor,W_hz:torch.Tensor,b_hz:torch.Tensor,W_in:torch.Tensor,b_in:torch.Tensor,W_hn:torch.Tensor,b_hn:torch.Tensor): 
        """
            the inputs shape is: [N,L,Hin]
            the h0 shape is: [N,Hout]
            the W_ir shape is: [Hout,Hin]
            the b_ir shape is: [Hout]
            the W_hr shape is: [Hout,Hout] 
            the b_hr shape is: [Hout]
            the W_iz shape is: [Hout,Hin]
            the b_iz shape is: [Hout]
            the W_hz shape is: [Hout,Hout]
            the b_hz shape is: [Hout]
            the W_in shape is: [Hout,Hin]
            the b_in shape is: [Hout]
            the W_hn shape is: [Hout,Hout]
            the b_hn shape is: [Hout] 
        """
        N,L,Hin = inputs.shape 
        _,Hout = h0.shape 
        outputs = torch.zeros(size=(N,L,Hout),device=inputs.device,dtype=inputs.dtype) 
        ###Lo primero de todo esto se va a tener que hacer
        for i in range(L): 
            #Primero calculamos el reset_gate: 
            reset_gate = sigmoid((W_ir @ inputs[:,i,:].transpose(0,1)).transpose(0,1) + b_ir + (W_hr @ outputs[:,i-1,:].transpose(0,1)).transpose(0,1) + b_hr) if i > 0 else sigmoid((W_ir @ inputs[:,i,:].transpose(0,1)).transpose(0,1) + b_ir + (W_hr @ h0.transpose(0,1)).transpose(0,1) + b_hr)
            #Ahora calculamos el update gate: 
            update_gate = sigmoid((W_iz @ inputs[:,i,:].transpose(0,1)).transpose(0,1) + b_iz + (W_hz @ outputs[:,i-1,:].transpose(0,1)).transpose(0,1) + b_hz) if i > 0 else sigmoid((W_iz @ inputs[:,i,:].transpose(0,1)).transpose(0,1) + b_iz + (W_hz @ h0.transpose(0,1)).transpose(0,1) + b_hz)
            #Ahora calculamos la candidate de la cell de memory: 
            nt = tanh((W_in @ inputs[:,i,:].transpose(0,1)).transpose(0,1) + b_in + reset_gate*((W_hn @ outputs[:,i-1,:].transpose(0,1)).transpose(0,1) + b_hn)) if i > 0 else tanh((W_in @ inputs[:,i,:].transpose(0,1)).transpose(0,1) + b_in + reset_gate*((W_hn @ h0.transpose(0,1)).transpose(0,1) + b_hn))
            #Ahora ya hacemos el computo del vector de estados:
            outputs[:,i,:] = (torch.ones_like(update_gate) - update_gate)*nt + update_gate*outputs[:,i-1,:] if i > 0 else (torch.ones_like(update_gate) - update_gate)*nt + update_gate*h0

        return outputs 
    







if __name__ == "__main__":
    torch.manual_seed(42)

    N, L, Hin, Hout = 2, 3, 4, 5

    x = torch.randn(N, L, Hin)         # input
    h0 = torch.randn(N, Hout)          # initial hidden state

    # Initializing weights and biases
    def init_weight(shape): return torch.randn(shape) * 0.1

    W_ir, b_ir = init_weight((Hout, Hin)), init_weight((Hout,))
    W_hr, b_hr = init_weight((Hout, Hout)), init_weight((Hout,))
    W_iz, b_iz = init_weight((Hout, Hin)), init_weight((Hout,))
    W_hz, b_hz = init_weight((Hout, Hout)), init_weight((Hout,))
    W_in, b_in = init_weight((Hout, Hin)), init_weight((Hout,))
    W_hn, b_hn = init_weight((Hout, Hout)), init_weight((Hout,))

    y = GRUFunction.apply(
        x, h0,
        W_ir, b_ir, W_hr, b_hr,
        W_iz, b_iz, W_hz, b_hz,
        W_in, b_in, W_hn, b_hn
    )

    print("Output GRU [N, L, Hout]:\n", y)
