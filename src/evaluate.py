import torch
import torch.nn.functional as nn


def evaluate(mixture, model, device, config):
    """
        mixture = batch, channel, num samples
    """
    with torch.no_grad():
        input_batch = mixture
        # mean, std
        if config.dset.norm == "z-score":
            mean_mixture = torch.mean(input_batch, dim=-1, keepdim=True)
            std_mixture = torch.std(input_batch, dim=-1, keepdim=True)
            input_batch = (input_batch-mean_mixture) / (std_mixture + 1e-6)
        
        if config.dset.norm == "linear-scale":
            max_mixture = torch.max(input_batch, dim=-1, keepdim=True).values
            min_mixture = torch.min(input_batch, dim=-1, keepdim=True).values
            input_batch = (input_batch-min_mixture) / (max_mixture - min_mixture + 1e-6)

        # segment 
        stride = config.model.win_length
        num_feature = int(config.dset.sample_rate*config.model.segment)
        input_batch = _prepare_input_wav_zero_filled(input_batch, num_feature, stride=stride)

        # merge segment, batch
        num_segment, batch, nchannel, nsample = input_batch.shape
        input_batch = input_batch.reshape(num_segment*batch, nchannel, nsample)
        if config.model.name in ("mel-rnn", "dcunet", "crn", "dnn", "unet"):
            input_batch = _stft(input_batch, config)

        # model
        if model:
            model.eval()
            # TODO, batch size > N -> RuntimeError: CUDA out of memory. 
            # Tried to allocate 2.64 GiB (GPU 0; 23.67 GiB total capacity; 6.86 GiB already allocated; 
            # 2.37 GiB free; 7.00 GiB reserved in total by PyTorch)            
            output = []
            segment = int(input_batch.shape[0]//2)
            
            input_batch_segment = input_batch[:segment].to(device)
            output_batch_segment = model(input_batch_segment)
            output.append(output_batch_segment)
            
            input_batch_segment = input_batch[segment:].to(device)
            output_batch_segment = model(input_batch_segment)
            output.append(output_batch_segment)
            
            # For torch 1.7.1, AttributeError: module 'torch' has no attribute 'concat'
            try: 
                output = torch.concat(output, dim=0)
            except AttributeError:
                output = torch.cat(output, dim=0)
        
            del input_batch_segment, output_batch_segment
        else:
            output = input_batch

        if config.model.name in ("mel-rnn", "dcunet", "crn", "dnn", "unet"):
            output = _istft(output, config)
        
        if config.model.name in ("conv-tasnet"):
            num_sources = len(config.model.sources)
            output = output.reshape(num_segment, batch, num_sources, nchannel, nsample)
            shape = list(mixture.shape)
            shape = shape[:-2] + [num_sources, shape[-2]] + [num_feature + stride*(output.shape[0]-1)]
        else:
            output = output.reshape(num_segment, batch, nchannel, nsample)
            shape = list(mixture.shape)
            shape = shape[:-1] + [num_feature + stride*(output.shape[0]-1)]
        
        enhanced = torch.zeros(size=shape, dtype=mixture.dtype)
        enhanced[..., :num_feature] = output[0, ...]
        for ibatch in range(output.shape[0]-1):
            curr_loc = num_feature + stride*ibatch
            enhanced[..., curr_loc: curr_loc+stride] = output[ibatch+1, ..., -stride:]

        enhanced = enhanced[..., :mixture.shape[-1]]        
        
        if config.dset.norm == "z-score":
            enhanced = enhanced*(std_mixture+1e-6) + mean_mixture
            
        if config.dset.norm == "linear-scale":
            enhanced = enhanced*(max_mixture-min_mixture+1e-6) + min_mixture

    return enhanced


def _stft(tensor: torch.Tensor, config):
    batch, nchannel, nsample = tensor.size()

    tensor = tensor.reshape(batch*nchannel, nsample)

    tensor = torch.stft(input=tensor,
                    n_fft=config.model.n_fft,
                    hop_length=config.model.hop_length,
                    win_length=config.model.win_length,
                    window=torch.hann_window(window_length=config.model.win_length, dtype=tensor.dtype, device=tensor.device),
                    center=config.model.center,
                    pad_mode="reflect",
                    normalized=False, # *frame_length**(-0.5)
                    onesided=None,
                    return_complex=False,
                    )
    tensor /= config.model.win_length
    _, nfeature, nframe, ndtype = tensor.size()
    tensor = tensor.reshape(batch, nchannel, nfeature, nframe, ndtype)
    return tensor

def _istft(tensor: torch.Tensor, config):
    batch, nchannel, nfeature, nframe, ndtype = tensor.size()
    tensor *= config.model.win_length
    tensor = tensor.reshape(batch*nchannel, nfeature, nframe, ndtype)
    tensor_complex = torch.complex(real=tensor[..., 0], imag=tensor[..., 1])
    
    tensor = torch.istft(
        input=tensor_complex,
        n_fft=config.model.n_fft,
        hop_length=config.model.hop_length,
        win_length=config.model.win_length,
        window=torch.hann_window(window_length=config.model.win_length, dtype=tensor.dtype, device=tensor.device),
        center=config.model.center,
        length=int(config.model.segment*config.model.sample_rate),
        normalized=False,
        onesided=None,
        return_complex=False,
    )
    _, nsample = tensor.size()
    tensor = tensor.reshape(batch, nchannel, nsample)
    return tensor

def _prepare_input_wav_zero_filled(wav, num_feature, stride):
    assert wav.shape[-1] >= num_feature, "the length of data is too short comparing the number of features..."
    if (wav.shape[-1] - num_feature) % stride != 0:
        npad = stride*((wav.shape[-1] - num_feature)//stride + 1) - (wav.shape[-1] - num_feature)
        padding = [0 for _ in range(len(wav.shape)*2)]
        padding[1] = npad
        wav_padded = nn.pad(wav, pad=padding, mode="constant", value=0.)
    else:
        wav_padded = wav

    num_segment = (wav_padded.shape[-1] - num_feature) // stride +1

    shape = list(wav.shape)
    shape = [num_segment] + shape[:-1] + [num_feature]
    wav_segments = torch.zeros(size=shape, dtype=wav.dtype)
    
    for index in range(num_segment):
        wav_segments[index, ...] = wav_padded[..., index*stride:index*stride+num_feature]
    
    return wav_segments