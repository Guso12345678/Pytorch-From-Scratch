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

        N,L,_ = inputs.shape 
        _,Hout = h0.shape
        h_estados = torch.empty(size=(N,L,Hout),dtype=h0.dtype, device=h0.device)
        c_estados = torch.empty(size=(N,L,Hout),dtype=c0.dtype, device=c0.device)

        for i in range(L): 
            h_prev = h_estados[:,i-1,:] if i > 0 else h0 
            c_prev = c_estados[:,i-1,:] if i > 0 else c0 

            gi = inputs[:,i,:] @ weights_ih.T + bias_ih # shape: [N,4*Hout]
            gh = h_prev @ weights_hh.T + bias_hh #shape: [N,4*Hout]

            ii = gi[:,:Hout]
            ih = gh[:,:Hout]
            it = sigmoid(ii + ih)

            fi = gi[:,Hout:2*Hout]
            fh = gh[:,Hout:2*Hout]
            ft = sigmoid(fi + fh)

            ggi = gi[:,2*Hout:3*Hout]
            ggh = gh[:,2*Hout:3*Hout]
            gt = tanh(ggi + ggh)

            oi = gi[:,3*Hout:]
            oh = gh[:,3*Hout:]
            ot = sigmoid(oi + oh)

            ct = ft*c_prev + it*gt 
            ht = ot*tanh(ct)

            c_estados[:,i,:] = ct
            h_estados[:,i,:] = ht 
        
        return torch.empty_like(h_estados),(h_estados[:,-1,:], c_estados[:,-1,:]) 
    

import torch
import torch.nn as nn

class LSTMTriDim(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.bias_ih = nn.Parameter(torch.zeros(4 * hidden_size))

        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(4 * hidden_size))

    def forward(self, x, h0=None, c0=None):
        N, L, Hin = x.shape
        Hout = self.hidden_size

        if h0 is None:
            h0 = torch.zeros(1, N, Hout, dtype=x.dtype, device=x.device)
        if c0 is None:
            c0 = torch.zeros(1, N, Hout, dtype=x.dtype, device=x.device)

        # quitar dimensión batch para compatibilidad con tu función
        h0_ = h0[0]
        c0_ = c0[0]

        h_last, c_last = LSTMTriDimFunct.apply(
            x, h0_, c0_,
            self.weight_ih, self.bias_ih,
            self.weight_hh, self.bias_hh
        )

        return h_last.unsqueeze(0), (h_last.unsqueeze(0), c_last.unsqueeze(0))



