import torch 

class CircularPad2DFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs,padding_size): 
        """
            the inputs shape is: [N,C,Hin,Win]
            the outputs shape is: [N,C,Hout,Wout]
        """
        N,C,Hin,Win = inputs.shape 
        p_left, p_right, p_top, p_bottom = padding_size

        Hout = Hin + p_top + p_bottom 
        Wout = Win + p_left + p_right 

        outputs = torch.empty(size=(N,C,Hout,Wout))
        outputs[:,:,p_top:p_top+Hin,:p_left] = inputs[:,:,:,-p_left:]
        outputs[:,:,p_top:p_top+Hin,p_left+Win:] = inputs[:,:,:,:p_right]

        outputs[:,:,:p_top,p_left:p_left+Win] = inputs[:,:,-p_top:,:]
        outputs[:,:,p_top+Hin:,p_left:p_left+Win] = inputs[:,:,:p_bottom,:]

        outputs[:,:,:p_top,:p_left] = inputs[:,:,-p_top:,-p_left:] 
        outputs[:,:,:p_top,p_left+Win:] = inputs[:,:,-p_top:,:p_right]
        outputs[:,:,p_top+Hin:,:p_left] = inputs[:,:,:p_bottom,-p_left:]
        outputs[:,:,p_top+Hin:,p_left+Win:] = inputs[:,:,:p_bottom,:p_right]

        outputs[:,:,p_top:p_top+Hin,p_left:p_left+Win] = inputs 
        return outputs  
