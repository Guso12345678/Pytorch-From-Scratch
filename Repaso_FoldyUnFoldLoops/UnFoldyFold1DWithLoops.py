import torch 
def unfold1DWithLoops(inputs, kernel_size, stride, dilation=1, padding=0):
    N, C, L_in = inputs.shape
    L_out = int(((L_in - dilation * (kernel_size - 1) - 1) / stride) + 1)

    outputs = torch.empty((N, C * kernel_size, L_out), dtype=inputs.dtype, device=inputs.device)

    for l in range(L_out):
        for k in range(kernel_size):
            idx = l * stride + k * dilation
            outputs[:, C * k:C * (k + 1), l] = inputs[:, :, idx]
    return outputs
 


def fold1DWithLoops(inputs, output_size, kernel_size, stride=1, dilation=1, padding=0):
    N, Ck, L_out = inputs.shape
    C = Ck // kernel_size

    outputs = torch.zeros((N, C, output_size), dtype=inputs.dtype, device=inputs.device)
    counts = torch.zeros_like(outputs)

    for l in range(L_out):
        for k in range(kernel_size):
            idx = l * stride + k * dilation
            if idx < output_size:
                outputs[:, :, idx] += inputs[:, C * k:C * (k + 1), l]
                counts[:, :, idx] += 1

    outputs = outputs / counts.clamp(min=1)
    return outputs

    