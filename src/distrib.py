import torch
from torch.utils.data import ConcatDataset, random_split
from torch.nn.functional import pad
# from torch.nn.utils.rnn import pad_sequence

from .dataset import WavDataset
from .utils import find_folder, obj2dict

from .model.conv_tasnet import ConvTasNet
from .model.crn import CRN
from .model.dccrn import DCCRN
from .model.dcunet import DCUnet
from .model.demucs import Demucs
from .model.wav_unet import WavUnet
from .model.mel_rnn import MelRNN

def collate_fn_pad(config, drop_last=True):
    def _collate_fn_pad(batch):
        mixture_list = []
        clean_list = []
        names = []
        index_batch = []

        for mixture, clean, name in batch:
            segment_length = int(config.segment * config.sample_rate)

            if drop_last and mixture.size()[-1] % segment_length != 0 and mixture.size()[-1] > segment_length:
                mixture = mixture[...,:segment_length*int(mixture.size()[-1]//segment_length)]
                clean = clean[...,:segment_length*int(mixture.size()[-1]//segment_length)]
            elif not drop_last and mixture.size()[-1] % segment_length != 0 and mixture.size()[-1] > segment_length:
                npad = int(mixture.size()[-1]//segment_length+1)*segment_length - mixture.size()[-1]
                mixture = pad(mixture, pad=(0, npad), mode="constant", value=0)
                clean = pad(clean, pad=(0, npad), mode="constant", value=0)
            elif mixture.size()[-1] < segment_length:
                npad = mixture.size()[-1] - segment_length
                mixture = pad(mixture, pad=(0, npad), mode="constant", value=0)
                clean = pad(clean, pad=(0, npad), mode="constant", value=0)

            channel, length = mixture.size()
            nsegment = int(length // segment_length)
            mixture = mixture.view(channel, nsegment, segment_length)
            clean = clean.view(channel, nsegment, segment_length)

            mixture_list.append(mixture) 
            clean_list.append(clean)
            names.append(name)
            index_batch.append(mixture.size()[1])

            assert mixture.size() == clean.size()

        # seq_list = [(T1, nsample), (T2, nsample), ...]
        #   item.size() must be (T, *)
        #   return (longest_T, len(seq_list), *)
        # list = pad_sequence(list)
        
        mixture_list = torch.concat(mixture_list, dim=1).permute(1, 0, 2)
        clean_list = torch.concat(clean_list, dim=1).permute(1, 0, 2)

        return mixture_list, clean_list, names, index_batch
    return _collate_fn_pad

def get_train_wav_dataset(config):
    mixture_dataset_path_list = find_folder(name="noisy_trainset", path=config.wav)
    clean_dataset_path_list = find_folder(name="clean_trainset", path=config.wav)
    
    dataset = []
    sample_length = int(config.sample_rate*config.segment)
    for mixture_path_dataset, clean_path_dataset in zip(mixture_dataset_path_list, clean_dataset_path_list):
        dataset.append(WavDataset(mixture_dataset=mixture_path_dataset,
                                clean_dataset=clean_path_dataset,
                                sample_length=sample_length if not config.use_all else None,
                                limit=None,
                                offset=0))
    dataset = ConcatDataset(dataset)

    n_train = int(len(dataset) * config.split)
    n_validation = len(dataset)- int(len(dataset) * config.split)
    
    train_dataset, validation_dataset = random_split(dataset, lengths=[n_train, n_validation])

    return train_dataset, validation_dataset


def get_model(config):
    klass = {
            'mel-rnn': MelRNN,
            'dccrn': DCCRN,
            'dcunet': DCUnet,
            'demucs': Demucs,
            'wav-unet': WavUnet,
            'conv-tasnet': ConvTasNet,
            'crn': CRN,
    }[config.name]

    kwargs = obj2dict(config)
    model = klass(**kwargs)
    return model
    
def get_optimizer(config, model):
    if config.optim == "sgd":
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            )
    elif config.optim == "adam":
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            )
    else:
        raise ValueError(f"Optimizer {config.optim} cannot use...")

    return optimizer

def get_loss_function(config):
    if config.loss == "l1": # mae
        loss_function = torch.nn.functional.l1_loss
    elif config.loss == "mse":
        loss_function = torch.nn.functional.mse_loss
    # elif config.loss == 'psa':
    #     from loss import phase_sensitive_approximate_loss
    #     loss_function = phase_sensitive_approximate_loss
    else:
        raise ValueError(f"Loss function {config.loss} cannot use...")

    return loss_function
