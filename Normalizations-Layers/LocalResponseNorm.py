import torch 
class LocalResponseNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, alpha, beta, k, size):
        """
        inputs: [N, C, H, W]
        """
        N, C, H, W = inputs.shape
        outputs = torch.zeros_like(inputs)

        half = size // 2
        squared_inputs = inputs.pow(2)

        for c in range(C):
            # Rango de canales vecinos
            start = max(0, c - half)
            end = min(C, c + half + 1)

            # Suma de cuadrados de canales vecinos
            scale = k + (alpha / size) * torch.sum(squared_inputs[:, start:end, :, :], dim=1, keepdim=True)

            # Normalización
            outputs[:, c, :, :] = inputs[:, c, :, :] / (scale.squeeze(1).pow(beta))

        ctx.save_for_backward(inputs, outputs, scale)
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.k = k
        ctx.size = size
        return outputs 
    
