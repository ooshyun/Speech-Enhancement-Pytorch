"""
sources (num_spk, num_channel, num_samples)
mixture (num_channel, num_samples)

"""
import os
import numpy as np
import glob
import torch
from torch.utils.data import (
    Dataset,
    ConcatDataset,
    Subset,
    DataLoader,
    random_split,
)
import omegaconf
from torch.nn.functional import pad
# from torch.nn.utils.rnn import pad_sequence

import typing as tp
from .dataset import WavDataset, ClarityWavDataset
from .utils import find_folder, obj2dict, split_list
from .loss import loss_sisdr

from .model.conv_tasnet import ConvTasNet
from .model.crn import CRN
from .model.dccrn import DCCRN
from .model.dcunet import DCUnet
from .model.demucs import Demucs
from .model.wav_unet import WavUnet
from .model.mel_rnn import MelRNN
from .model.dnn import DeepNeuralNetwork
from .model.unet import UNet

def collate_fn_pad(config, drop_last=True):
    def _collate_fn_pad(batch):
        batch_mixture = []
        mixture_metadata_list = []

        batch_sources = []
        sources_metadata_list = []

        names = []
        index_batch = []

        for one_batch in batch:
            mixture, sources, mixture_metadata, sources_metadata, name = one_batch
      
            mixture_metadata_list.append(mixture_metadata)
            sources_metadata_list.append(sources_metadata)
            
            segment_length = int(config.segment * config.sample_rate)

            if drop_last and mixture.size()[-1] % segment_length != 0 and mixture.size()[-1] > segment_length:
                mixture = mixture[...,:segment_length*int(mixture.size()[-1]//segment_length)]
                sources = sources[...,:segment_length*int(mixture.size()[-1]//segment_length)]
            
            elif not drop_last and mixture.size()[-1] % segment_length != 0 and mixture.size()[-1] > segment_length:
                npad = int(mixture.size()[-1]//segment_length+1)*segment_length - mixture.size()[-1]
                mixture = pad(mixture, pad=(0, npad), mode="constant", value=0)
                sources = pad(sources, pad=(0, npad), mode="constant", value=0)
            
            elif mixture.size()[-1] < segment_length:
                npad = mixture.size()[-1] - segment_length
                mixture = pad(mixture, pad=(0, npad), mode="constant", value=0)
                sources = pad(sources, pad=(0, npad), mode="constant", value=0)
            
            num_channel, length = mixture.size()
            nsegment = int(length // segment_length)
            mixture = mixture.view(num_channel, nsegment, segment_length)

            num_spk = sources.size()[0]
            sources = sources.view(num_spk, num_channel, nsegment, segment_length)
            
            batch_mixture.append(mixture) 
            batch_sources.append(sources)
            
            names.append(name)
            index_batch.append(mixture.size()[1])
            
            assert mixture.size() == sources.size()[1:]

        # seq_list = [(T1, nsample), (T2, nsample), ...]
        #   item.size() must be (T, *)
        #   return (longest_T, len(seq_list), *)
        # list = pad_sequence(list)

        try:
            batch_mixture = torch.concat(batch_mixture, dim=1).permute(1, 0, 2)
            batch_sources = torch.concat(batch_sources, dim=2).permute(2, 0, 1, 3)
        except AttributeError:
            # For torch 1.7.1, AttributeError: module 'torch' has no attribute 'concat'
            batch_mixture = torch.cat(batch_mixture, dim=1).permute(1, 0, 2)
            batch_sources = torch.cat(batch_sources, dim=2).permute(2, 0, 1, 3)
        return batch_mixture, batch_sources, mixture_metadata_list, sources_metadata_list, names, index_batch
    return _collate_fn_pad

def get_train_wav_voicebankdemand(config):
    sample_length = int(config.sample_rate*config.segment)
    mixture_dataset_path_list = find_folder(name="noisy_trainset", path=config.wav)
    clean_dataset_path_list = find_folder(name="clean_trainset", path=config.wav)    
    train_dataset = []
    test_dataset = []

    for mixture_path_dataset, clean_path_dataset in zip(mixture_dataset_path_list, clean_dataset_path_list):
        num_files = len(glob.glob(f"{mixture_path_dataset}/*.wav"))
        assert num_files==len(glob.glob(f"{clean_path_dataset}/*.wav")), f"The number of clean and mixture files should be same..."
        
        scene_train_list, scene_test_list = split_list(np.arange(num_files), ratio=config.split)
        train_dataset.append(WavDataset(mixture_dataset=mixture_path_dataset,
                                        clean_dataset=clean_path_dataset,
                                        scenes= scene_train_list,
                                        sample_length=sample_length if not config.use_all else None,
                                        limit=None,
                                        offset=0,
                                        normalize=config.norm,
                                        sample_rate=config.sample_rate,
                                        audio_channels=config.audio_channels,
                                        train=True))

        test_dataset.append(WavDataset(mixture_dataset=mixture_path_dataset,
                                        clean_dataset=clean_path_dataset,
                                        scenes=scene_test_list,
                                        sample_length=sample_length if not config.use_all else None,
                                        limit=None,
                                        offset=0,
                                        normalize=config.norm,
                                        sample_rate=config.sample_rate,
                                        audio_channels=config.audio_channels,
                                        train=False))

    train_dataset = ConcatDataset(train_dataset)
    test_dataset = ConcatDataset(test_dataset)

    ratio_train = config.split[0]
    n_train = int(len(train_dataset) * ratio_train)
    n_validation = len(train_dataset)- n_train
    
    train_dataset, validation_dataset = random_split(train_dataset, lengths=[n_train, n_validation])

    print(f"Train {len(train_dataset)}, Validation {len(validation_dataset)}, Test {len(test_dataset)}")

    return train_dataset, validation_dataset, test_dataset

def get_train_wav_clarity(config):
    sample_length = int(config.sample_rate*config.segment)

    scene_list = list(omegaconf.OmegaConf.load(os.path.join(config.wav, "custom_metadata/scenes.train.scene_name.json")))
    scene_train_list, scene_test_list = split_list(scene_list, ratio=config.split)

    train_dataset = ClarityWavDataset(path_dataset=config.wav,
                                    scenes= scene_train_list,
                                    sample_length=sample_length if not config.use_all else None,
                                    limit=None,
                                    offset=0,
                                    normalize=config.norm,
                                    sample_rate=config.sample_rate,
                                    audio_channels=config.audio_channels,                                
                                    train=True)     
    
    ratio_train = config.split[0]
    n_train = int(len(train_dataset) * ratio_train)
    n_validation = len(train_dataset)- int(len(train_dataset) * ratio_train)
    
    train_dataset, validation_dataset = random_split(train_dataset, lengths=[n_train, n_validation])

    test_dataset = ClarityWavDataset(path_dataset=config.wav,
                                        scenes=scene_test_list,
                                        sample_length=None,
                                        limit=None,
                                        offset=0,
                                        normalize=config.norm,
                                        sample_rate=config.sample_rate,
                                        audio_channels=config.audio_channels,
                                        train=False) 

    print(f"Train {len(train_dataset)}, Validation {len(validation_dataset)}, Test {len(test_dataset)}")

    return train_dataset, validation_dataset, test_dataset

def get_dev_wav_clarity(config):
    sample_length = int(config.sample_rate*config.segment)

    scene_list = list(omegaconf.OmegaConf.load(os.path.join(config.wav, "custom_metadata/scenes.dev.scene_name.json")))
    dev_dataset = ClarityWavDataset(path_dataset=config.wav,
                                    scenes= scene_list,
                                    sample_length=sample_length if not config.use_all else None,
                                    limit=None,
                                    offset=0,
                                    normalize=config.norm,
                                    sample_rate=config.sample_rate,
                                    audio_channels=config.audio_channels,
                                    train=False,
                                    dev_clarity=True)     
    return dev_dataset

def get_train_wav_dataset(config):
    if config.name == "VoiceBankDEMAND":
        train_dataset, validation_dataset, test_dataset = get_train_wav_voicebankdemand(config)
    elif config.name == "Clarity":
        train_dataset, validation_dataset, test_dataset = get_train_wav_clarity(config)
    else:
        raise ValueError(f"{config.name} dataset is not implemented")

    return train_dataset, validation_dataset, test_dataset


def get_dataloader(datasets: tp.List[Dataset], config, train=True) -> tp.List[DataLoader]:
    dataloaders = []
    for dset in datasets:
        dataloaders.append(DataLoader(dataset=dset,
                                batch_size=config.solver.batch_size if train else 1,
                                shuffle=True,
                                num_workers=config.solver.num_workers,
                                collate_fn=collate_fn_pad(config.dset, drop_last=True) if train else None,
                                pin_memory=True,
                                # drop_last=True,
                                prefetch_factor=2,
                                ))
    return dataloaders


def get_model(config):
    klass = {
            'dnn': DeepNeuralNetwork,
            'mel-rnn': MelRNN,
            'unet': UNet,
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
    elif config.loss == "si-sdr":
        loss_function = loss_sisdr
    # elif config.loss == 'psa':
    #     from loss import phase_sensitive_approximate_loss
    #     loss_function = phase_sensitive_approximate_loss
    else:
        raise ValueError(f"Loss function {config.loss} cannot use...")

    return loss_function
