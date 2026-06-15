import torch 

class ReflectionPad2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, padding_size):
        """
        inputs: [N,C,Hin,Win]
        padding_size = (pleft, pright, ptop, pbottom)
        """
        N, C, Hin, Win = inputs.shape
        pleft, pright, ptop, pbottom = padding_size
        Hout = Hin + ptop + pbottom
        Wout = Win + pleft + pright

        outputs = torch.empty((N, C, Hout, Wout), device=inputs.device, dtype=inputs.dtype)

        # Centro
        outputs[:, :, ptop:ptop+Hin, pleft:pleft+Win] = inputs

        # Izquierda
        outputs[:, :, ptop:ptop+Hin, :pleft] = torch.flip(inputs[:, :, :, 1:pleft+1], dims=[-1])

        # Derecha
        outputs[:, :, ptop:ptop+Hin, pleft+Win:] = torch.flip(inputs[:, :, :, -pright-1:-1], dims=[-1])

        # Arriba
        outputs[:, :, :ptop, :] = torch.flip(outputs[:, :, ptop+1:ptop+ptop+1, :], dims=[-2])

        # Abajo
        outputs[:, :, ptop+Hin:, :] = torch.flip(outputs[:, :, ptop+Hin-2:ptop+Hin-2-pbottom:-1, :], dims=[-2])

        ctx.input_shape = inputs.shape
        ctx.padding_size = padding_size
        return outputs
