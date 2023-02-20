import torch
import numpy as np
import torch.nn as nn
from itertools import product, permutations

try:
    from clarity.enhancer.compressor import CompressorTorch
    from clarity.enhancer.nalr import NALRTorch
    LIB_CLARITY = True
except ModuleNotFoundError:
    print("There's no clarity library")
    LIB_CLARITY = False

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


def loss_phase_sensitive_spectral_approximation(enhance, target, mixture):
    """
    - Erdogan, Hakan, et al. "Phase-sensitive and recognition-boosted speech separation using deep recurrent neural networks." 
    2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2015.

        Despite the use of the MSA/PSA objective function, 
        it is still desirable to have the output of the network be a mask, 
        since then the entire dynamic range of the data does not have to be covered by the output of the network. 
        It is convenient in this context to truncate a to between 0 and 1, to fit the range of a sigmoid unit. 
        In addition, we conjecture that mask prediction would also avoid global variance problems reported in [10].

        [10] “Global variance equalization for improving deep neural network based speech enhancement,” in Proc. of ICASSP, 
        Florence, Italy, 2014.    
    """
    eps = nn.Parameter(data=torch.ones((1, ), dtype=torch.float32)*1e-9, requires_grad=False).to(enhance.device)
    
    angle_mixture = torch.tanh(mixture[..., 1] / (mixture[..., 0]+eps))
    angle_target = torch.tanh(target[..., 1] / (target[..., 0]+eps))
    
    amplitude_enhance = torch.sqrt(enhance[..., 1]**2 + enhance[..., 0]**2)
    amplitude_target = torch.sqrt(target[..., 1]**2 + target[..., 0]**2)

    loss = amplitude_enhance - amplitude_target*torch.cos(angle_target-angle_mixture)
    loss = torch.mean(loss**2) # mse
    return loss

def UtterenceBaasedPermutationInvariantTraining(enhance, target: list, loss_function, mixture=None, return_comb=False):
    # O(S^2), S= the number of speaker    
    assert enhance.shape == target.shape, f"enhance and target shape did not match...{enhance.shape}, {target.shape}"

    nspk_enhance = enhance.shape[1]
    nspk_target = target.shape[1]

    id_spks_enhance = list(range(nspk_enhance))
    id_spks_target = list(range(nspk_target))

    product_id_spks = product(id_spks_enhance, id_spks_target )
    loss_id_spks = torch.zeros(size=(nspk_enhance, nspk_target), dtype=torch.float32, requires_grad=False)
    with torch.no_grad():
        for i, (ienhance, itarget) in enumerate(product_id_spks):
            loss_id_spks[ienhance, itarget] = loss_function(enhance[:, ienhance, ...], target[:, itarget, ...]) if mixture is None else loss_function(enhance[:, ienhance, ...], target[:, itarget, ...], mixture) 
        
        combinations_id_spks_enhance = permutations(id_spks_enhance)

        combination = None
        loss_min = 1e9
        for ienhance in combinations_id_spks_enhance:
            loss = 0    
            comb = []
            for itarget in id_spks_target:
                loss += loss_id_spks[ienhance[itarget], itarget]
                comb.append((ienhance[itarget], itarget))

            if loss_min > loss:
                combination = comb
                loss_min = loss

    loss = torch.Tensor(torch.zeros(size=(1, )))
    loss = loss.to(target.device)
    for ienhance, itarget in combination:
        if mixture is None:
            loss += loss_function(enhance[:, ienhance, ...], target[:, itarget, ...]) 
        else:
            loss += loss_function(enhance[:, ienhance, ...], target[:, itarget, ...]) 
    loss /= nspk_enhance
    if not return_comb:
        return loss
    else:
        return loss, combination