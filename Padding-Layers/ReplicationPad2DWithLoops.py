import torch

class ReplicationPad2DWithLoops(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, padding_size):
        """
        inputs: shape [N, C, H, W]
        padding_size: (pleft, pright, ptop, pbottom)
        """
        N, C, H, W = inputs.shape
        pleft, pright, ptop, pbottom = padding_size
        H_out = H + ptop + pbottom
        W_out = W + pleft + pright

        outputs = torch.empty((N, C, H_out, W_out), device=inputs.device, dtype=inputs.dtype)

        # BUCLE CENTRAL: copiar el input original al centro del output
        for n in range(N):
            for c in range(C):
                for i in range(H):
                    for j in range(W):
                        outputs[n, c, ptop + i, pleft + j] = inputs[n, c, i, j]

        # BUCLE IZQUIERDA
        for n in range(N):
            for c in range(C):
                for i in range(H):
                    for j in range(pleft):
                        outputs[n, c, ptop + i, j] = inputs[n, c, i, 0]

        # BUCLE DERECHA
        for n in range(N):
            for c in range(C):
                for i in range(H):
                    for j in range(pright):
                        outputs[n, c, ptop + i, pleft + W + j] = inputs[n, c, i, -1]

        # BUCLE ARRIBA
        for n in range(N):
            for c in range(C):
                for i in range(ptop):
                    for j in range(W_out):
                        outputs[n, c, i, j] = outputs[n, c, ptop, j]

        # BUCLE ABAJO
        for n in range(N):
            for c in range(C):
                for i in range(pbottom):
                    for j in range(W_out):
                        outputs[n, c, ptop + H + i, j] = outputs[n, c, ptop + H - 1, j]

        ctx.input_shape = inputs.shape
        ctx.padding_size = padding_size
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        pleft, pright, ptop, pbottom = ctx.padding_size
        _, _, H, W = ctx.input_shape
        grad_inputs = grad_outputs[:, :, ptop:ptop + H, pleft:pleft + W]
        return grad_inputs, None
