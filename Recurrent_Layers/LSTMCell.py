import torch 

def sigmoid(x):
    return 1/(1 + torch.exp(-x)) 
def tanh(x): 
    return (torch.exp(x) - torch.exp(-x))/(torch.exp(x) + torch.exp(-x))
class LSTMCellFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs,hx,cx,weight_ih,weight_hh,bias_ih,bias_hh):
        """
            inputs shape is: [bacth_size, input_size]
            hx=ht-1 shape is: [batch_size,hidden_size]
            cx=ct-1 shape is: [batch_size,hidden_size]
            weight_ih shape is: [4*hidden_size,input_size]
            weight_hh shape is: [4*hidden_size,hidden_size]
            bias_ih shape is: [4*hidden_size]
            bias_hh shape is: [4*hidden_size]
        """ 

        hidden_size = int(hx.shape[1]) 
        gi = inputs @ weight_ih.T + bias_ih # Shape [batch_size,4*hidden_size] 
        gh = hx @ weight_hh.T + bias_hh #Shape [batch_size,4*hidden_size]

        ii = gi[:,:hidden_size] #Shape [batch_size,hidden_size]
        ih = gh[:,:hidden_size] #Shape [batch_size,hidden_size]
        it = sigmoid(ii + ih) #Puerta de entrada 

        fi = gi[:,hidden_size:2*hidden_size]
        fh = gh[:,hidden_size:2*hidden_size]
        ft = sigmoid(fh + fi)

        ggi = gi[:,2*hidden_size:3*hidden_size]
        ggh = gh[:,2*hidden_size:3*hidden_size]
        gg = tanh(ggi + ggh)

        oi = gi[:,3*hidden_size:]
        oh = gh[:,3*hidden_size:]
        ot = sigmoid(oi+oh)

        ct = ft*cx + it*gg
        ht = ot*tanh(ct)
        return ht, ct
    


