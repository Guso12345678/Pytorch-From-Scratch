import torch 

def sigmoid(x): 
    return 1/(1 + torch.exp(-x))

def tanh(x): 
    return (torch.exp(x)- torch.exp(-x))/(torch.exp(x)+torch.exp(-x))
class GRUCellFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs,hx,weight_i,weight_h,bias_h,bias_i): 
        """
            the inputs shape is: [batch_size,input_size]
            the hx shape is: [batch_size,hidden_size]
            the weight_i shape is: [3*hidden_size,input_size]
            the weight_h shape is: [3*hidden_size,hidden_size]
            the bias_i shape is: [3*hidden_size]
            the bias_h shape is: [3*hidden_size]
        """
        gi = inputs @ weight_i.T + bias_i 
        gh = hx @ weight_h.T + bias_h 

        hidden_size = int(hx.shape[1]) 

        #Reset Gate: 
        ri = gi[:,:hidden_size]
        rh = gh[:,:hidden_size]
        rt = sigmoid(ri + rh)

        #Udpate Gate: 
        zi = gi[:,hidden_size:hidden_size*2]
        zh = gh[:,hidden_size:hidden_size*2]
        zt = sigmoid(zi+zh)

        #Candidate Cell: 
        ni = gi[:,2*hidden_size:]
        nh = gh[:,2*hidden_size:]
        nt = tanh(ni + rt*nh)

        #State vector: 
        h = (1 - zt)*nt + zt*hx

        return h  

