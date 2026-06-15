import torch 

def norm_distance(p,tensor1,tensor2): 
    if p == 1: 
        return torch.abs(tensor1 - tensor2)
    else: 
        return torch.sqrt((tensor1 - tensor2)**2)  
def gradiente_normas(p,tensor1,tensor2): 
    if p == 1: 
        return torch.sign(tensor1 - tensor2)
    else:
        diff = tensor1 - tensor2
        norm = torch.sqrt((diff ** 2).sum(dim=1, keepdim=True))
        return diff / norm 

class TripletMarginLossFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,inputs1,inputs2,inputs3,eps,margin,p): 
        ctx.save_for_backward(inputs1,inputs2,inputs3)
        ctx.p = p 
        ctx.margin = margin 
        mask1 = norm_distance(p,inputs1,inputs2) - norm_distance(p,inputs1,inputs3) + margin > 0 
        mask2 = norm_distance(p,inputs1,inputs2) - norm_distance(p,inputs1,inputs3) + margin <= 0
        loss = torch.zeros_like(inputs1)
        ctx.mask1 = mask1
        loss[mask1] = norm_distance(p,inputs1[mask1],inputs2[mask1]) - norm_distance(p,inputs1[mask1],inputs3[mask1]) + margin 
        loss[mask2] = 0 
        return loss 
    @staticmethod
    def backward(ctx, grad_outputs):
        inputs1, inputs2, inputs3 = ctx.saved_tensors
        p = ctx.p 
        margin = ctx.margin 
        mask = ctx.mask  # (N, 1)

        grad_anchor = torch.zeros_like(inputs1)
        grad_pos = torch.zeros_like(inputs2)
        grad_neg = torch.zeros_like(inputs3)

        if p == 1:
            d_ap = torch.sign(inputs1 - inputs2)
            d_an = torch.sign(inputs1 - inputs3)
        else:
            d_ap = gradiente_normas(p, inputs1, inputs2)
            d_an = gradiente_normas(p, inputs1, inputs3)

        grad_anchor[mask.squeeze()] = d_ap[mask.squeeze()] - d_an[mask.squeeze()] #Se hacen la resta de ambos porque se tienen los dos en cuenta. 
        grad_pos[mask.squeeze()] = -d_ap[mask.squeeze()]
        grad_neg[mask.squeeze()] = d_an[mask.squeeze()]

        grad_anchor = grad_outputs * grad_anchor
        grad_pos = grad_outputs * grad_pos
        grad_neg = grad_outputs * grad_neg

        return grad_anchor, grad_pos, grad_neg, None, None, None

