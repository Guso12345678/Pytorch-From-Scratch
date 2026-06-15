import torch 

def unfold2DwithLoops(inputs,kernel_size,dilation,stride): 
    N,C,H,W = inputs.shape 
    kh,kw = kernel_size 
    dh,dw = dilation 
    sh,sw = stride 
    L_out_h = int(((H + 2*0 - dh*(kh - 1) -1 )/sh) + 1)
    L_out_w = int(((W + 2*0 - dw*(kw - 1) - 1)/sw) + 1)
    L_out = L_out_h*L_out_w 
    outputs = torch.empty(size=(N,C*kh*kw,L_out)) 
    for l in range(L_out): 
        i = l // L_out_w 
        j = l % L_out_w 
        h_start = i*sh 
        w_start = j*sw 
        for k in range(kh*kw): 
            Kh = k // kw 
            Kw = k % kw 
            hend = Kh*dh 
            wend = Kw*dw 
            h = h_start + hend 
            w = w_start + wend 

            outputs[:,k*C:C*(k+1),l] = inputs[:,:,h,w]

def fold2Dwithloops(inputs,output_size,kernel_size,dilation,stride): 
    N,Cykyk,Lout = inputs.shape 
    Kh,Kw = kernel_size
    K = Kh*Kw
    H_in,W_in = output_size 
    C = Cykyk // K 
    outputs = torch.zeros(size=(N,C,H_in,W_in))
    Dh,Dw = dilation 
    Sh, Sw = stride 
    H_out = (H_in -  Dh* (Kh - 1) - 1) // Sh + 1
    W_out = (W_in - Dw * (Kw - 1) - 1) // Sw + 1

    count = torch.zeros_like(outputs)

    for l in range(Lout): 
        i = l // W_out 
        j = l % W_out 
        h_start = i*Sh 
        w_start = j*Sh 
        for k in range(K): 
            kh = k // Kw 
            kw = k % Kw 
            h = h_start + kh*Dh 
            w = w_start + kw*Dw 
            if h < H_out and w < W_out:
                outputs[:,:,h,w] += inputs[:,:,C*k:C*(k+1),l]
                count[:,:,h,w] += 1 
    return outputs / torch.clamp(count,min=1)



