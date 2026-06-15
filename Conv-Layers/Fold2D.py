import torch 
class Fold2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, output_size, kernel_size, stride, dilation):
        """
        inputs: shape (N, C * Kh * Kw, L_out)
        output_size: (H, W) - tamaño de la imagen reconstruida
        kernel_size: (Kh, Kw)
        stride: (Sh, Sw)
        dilation: (Dh, Dw)
        """
        N, CK, L_out = inputs.shape
        H, W = output_size
        Kh, Kw = kernel_size
        Sh, Sw = stride
        Dh, Dw = dilation

        K = Kh * Kw
        C = CK // K

        H_out = (H - Dh * (Kh - 1) - 1) // Sh + 1
        W_out = (W - Dw * (Kw - 1) - 1) // Sw + 1

        output = torch.zeros((N, C, H, W), device=inputs.device, dtype=inputs.dtype)

        for l in range(L_out):         # l = ventana (columna en inputs)
            i = l // W_out             # fila en mapa de ventanas
            j = l % W_out              # columna en mapa de ventanas
            h_start = i * Sh
            w_start = j * Sw

            for k in range(K):         # posición dentro del kernel
                kh = k // Kw
                kw = k % Kw
                h = h_start + kh * Dh
                w = w_start + kw * Dw

                output[:, :, h, w] += inputs[:, k::K, l]

        # Guardar para backward (solo shape y parámetros necesarios)
        ctx.save_for_backward(torch.tensor([C, H, W]))
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.dilation = dilation

        return output
    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor):
        """
        grad_outputs: shape (N, C, H, W)
        returns: grad_inputs: shape (N, C * Kh * Kw, L_out)
        """
        (shape_tensor,) = ctx.saved_tensors
        C, H, W = shape_tensor.tolist()
        Kh, Kw = ctx.kernel_size
        Sh, Sw = ctx.stride
        Dh, Dw = ctx.dilation

        H_out = (H - Dh * (Kh - 1) - 1) // Sh + 1
        W_out = (W - Dw * (Kw - 1) - 1) // Sw + 1
        L_out = H_out * W_out
        K = Kh * Kw

        grad_inputs = torch.zeros((grad_outputs.size(0), C * K, L_out), device=grad_outputs.device, dtype=grad_outputs.dtype)

        for l in range(L_out):
            i = l // W_out
            j = l % W_out
            h_start = i * Sh
            w_start = j * Sw

            for k in range(K):
                kh = k // Kw
                kw = k % Kw
                h = h_start + kh * Dh
                w = w_start + kw * Dw

                grad_inputs[:, k::K, l] = grad_outputs[:, :, h, w]

        return grad_inputs, None, None, None, None

