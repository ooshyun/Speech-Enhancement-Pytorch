import torch

def loss_sisdr(inputs, targets):
    return -si_snr(inputs, targets)

def l2_norm(s1, s2):
    norm = torch.sum(s1*s2, -1, keepdim=True)
    return norm 

def si_snr(s1, s2, eps=1e-8):
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target =  s1_s2_norm/(s2_s2_norm+eps)*s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)
    return torch.mean(snr)

def PermutationInvariantTraining(enhance, target: list, loss_function):
    # O(S^2), S= the number of speaker    
    min_loss = 1e12    
    index_enhance, index_target = 0, 0
    for i in range(enhance.shape[1]):
        for j in range(target.shape[1]):
            loss = loss_function(enhance[:, i, ...], target[:, j, ...])
            if loss < min_loss:
                min_loss = loss
                index_enhance, index_target = i, j

    return index_enhance, index_target

