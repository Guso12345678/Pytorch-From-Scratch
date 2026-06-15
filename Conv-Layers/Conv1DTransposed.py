import torch 
 
class Conv1DTransposeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, kernel, padding, dilation, stride):
        """
        inputs: (N, Cin, Lin)
        kernel: (Cin, Cout, K)
        returns: (N, Cout, Lout)
        """
        N, Cin, Lin = inputs.shape
        Cin_k, Cout, K = kernel.shape

        # Paso 1: expandir el input con ceros entre valores (stride)
        Lin_expanded = (Lin - 1) * stride + 1
        expanded = torch.zeros((N, Cin, Lin_expanded), device=inputs.device, dtype=inputs.dtype)
        expanded[:, :, ::stride] = inputs  # Inserta los valores reales

        # Paso 2: padding a izquierda y derecha
        total_pad = dilation * (K - 1)
        pad_left = total_pad - padding
        pad_right = total_pad - pad_left
        padded = torch.nn.functional.pad(expanded, (pad_left, pad_right))

        # Paso 3: unfold sobre la dimensión expandida
        # padded: (N, Cin, L_padded)
        # unfold → (N, Cin * K, L_out)
        unfolded = padded.unfold(dimension=2, size=K, step=1)  # (N, Cin, L_out, K)
        N, Cin, L_out, K = unfolded.shape
        unfolded = unfolded.permute(0, 2, 1, 3).reshape(N, L_out, Cin * K).transpose(1, 2)  # (N, Cin*K, L_out)

        # Paso 4: aplicar el kernel transpuesto como matmul
        kernel_t = kernel.transpose(0, 1).reshape(Cout, -1)  # (Cout, Cin*K)
        output = kernel_t @ unfolded  # (Cout, N, L_out)
        output = output.permute(1, 0, 2)  # (N, Cout, L_out)

        # Guardamos para el backward si lo necesitas
        ctx.save_for_backward(inputs, kernel)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation

        return output
    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, kernel = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation

        N, Cin, Lin = inputs.shape
        Cin_k, Cout, K = kernel.shape
        N_go, Cout_go, Lout = grad_outputs.shape
        assert Cout == Cout_go

        # === Gradiente con respecto a inputs ===
        # Expand grad_outputs as if it were the input to a Conv1D
        grad_expanded = torch.zeros((N, Cout, (Lout - 1) // 1 + 1), device=grad_outputs.device)
        grad_expanded[:, :, :] = grad_outputs

        # Kernel needs to be flipped and transposed to apply direct conv
        kernel_flip = kernel.transpose(0, 1).flip(dims=[2])  # (Cout, Cin, K)

        grad_inputs = torch.zeros_like(inputs)

        for n in range(N):
            for cin in range(Cin):
                for lin in range(Lin):
                    acc = 0.
                    for cout in range(Cout):
                        for k in range(K):
                            out_pos = lin * stride - padding + k * dilation
                            if 0 <= out_pos < Lout:
                                acc += grad_outputs[n, cout, out_pos] * kernel[cin, cout, k]
                    grad_inputs[n, cin, lin] = acc

        # === Gradiente con respecto al kernel ===
        grad_kernel = torch.zeros_like(kernel)

        # Expand inputs como hicimos en el forward
        Lin_expanded = (Lin - 1) * stride + 1
        expanded = torch.zeros((N, Cin, Lin_expanded), device=inputs.device)
        expanded[:, :, ::stride] = inputs
        total_pad = dilation * (K - 1)
        pad_left = total_pad - padding
        pad_right = total_pad - pad_left
        padded = torch.nn.functional.pad(expanded, (pad_left, pad_right))

        # Unfold grad_outputs como en forward
        unfolded = padded.unfold(dimension=2, size=K, step=1)  # (N, Cin, L_out, K)
        unfolded = unfolded.permute(0, 2, 1, 3)  # (N, L_out, Cin, K)

        for cin in range(Cin):
            for cout in range(Cout):
                for k in range(K):
                    # Sum across batch and spatial dim
                    val = 0.
                    for n in range(N):
                        for l in range(Lout):
                            val += inputs[n, cin, l // stride] * grad_outputs[n, cout, l] if l % stride == 0 else 0
                    grad_kernel[cin, cout, k] = val

        return grad_inputs, grad_kernel, None, None, None

