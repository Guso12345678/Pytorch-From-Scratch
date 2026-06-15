import torch 

class MaxOutFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs,k,d):
        """
            Los inputs son de shape [N,k*d]
        """ 
        inputs = inputs.view(inputs.size(0),d,k) #Ahora tiene un shape de: [N,k,d]
        N,D,K = inputs.shape 
        mask = torch.zeros_like(inputs)
        max_vals = torch.empty((N,D))
        for i in range(N): 
            for j in range(D): 
                m = inputs[i,j,0]
                mask_k = 0 
                for k in range(1,K): 
                    if inputs[i,j,k] > m: 
                        m = inputs[i,j,k]
                        mask_k = k 
                mask[i,j,mask_k] = 1
                max_vals[i,j] = m
        ctx.save_for_backward(inputs,mask)
        ctx.k = k 
        ctx.d = d 
        return max_vals 
