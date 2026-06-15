import torch

class CTCLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probs, targets, input_lengths, target_lengths, blank=0):
        """
        log_probs: (T, N, C) - log softmax probs
        targets: (sum(target_lengths),) - concatenated targets
        input_lengths: (N,)
        target_lengths: (N,)
        """
        T, N, C = log_probs.shape
        assert N == 1, "Esta implementación solo admite batch size = 1 para simplicidad"
        
        log_probs = log_probs.detach().requires_grad_()
        ctx.save_for_backward(log_probs, targets, input_lengths, target_lengths)
        ctx.blank = blank
        loss = 0.0
        total_len = target_lengths[0]
        input_len = input_lengths[0]
        for t in range(total_len):
            target_class = targets[t]
            logp = log_probs[t, 0, target_class]
            loss -= logp

        loss = loss / total_len

        return torch.tensor(loss, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        log_probs, targets, input_lengths, target_lengths = ctx.saved_tensors
        blank = ctx.blank
        T, N, C = log_probs.shape
        grad = torch.zeros_like(log_probs)
        total_len = target_lengths[0]
        input_len = input_lengths[0]

        for t in range(total_len):
            target_class = targets[t]
            grad[t, 0, target_class] = -1.0 / total_len
        grad *= grad_output

        return grad, None, None, None, None
