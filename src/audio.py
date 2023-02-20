import torch
import numpy as np
from .ha.compressor import CompressorTorch
from .ha.amplifier import NALRTorch

def convert_audio_channels(wav, channels=2):
    """Convert audio to the given number of channels.
        From. https://github.com/facebookresearch/denoiser
    """
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, but the stream have multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file have
        # one single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file have
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav

def amplify_torch(signal: torch.Tensor, 
            enhancer: NALRTorch,
            compressor: CompressorTorch,
            audiogram,
            soft_clip=True):
    """
        signal: batch, nspk, nchannel(stereo), nsample
    """

    cfs = np.array(audiogram["audiogram_cfs"])
    audiogram = np.array([audiogram[f"audiogram_levels_l"], 
                        audiogram[f"audiogram_levels_r"]])

    nalr_fir_left_torch = enhancer.build(audiogram[0], cfs)
    nalr_fir_right_torch = enhancer.build(audiogram[1], cfs)
    nalr_fir_left_torch = nalr_fir_left_torch.to(signal.device)
    nalr_fir_right_torch = nalr_fir_left_torch.to(signal.device)
    
    out_l_torch = enhancer.apply(nalr_fir_left_torch, signal[:, :, 0, ...])
    out_r_torch = enhancer.apply(nalr_fir_right_torch, signal[:, :, 1, ...])

    out_l_torch = compressor.process(out_l_torch)
    out_r_torch = compressor.process(out_r_torch)

    if soft_clip:
        out_l_torch = torch.tanh(out_l_torch)
        out_r_torch = torch.tanh(out_r_torch)

    return torch.stack([out_l_torch, out_r_torch], dim=2)
