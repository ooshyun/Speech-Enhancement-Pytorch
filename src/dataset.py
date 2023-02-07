import os
import time
import torch
import random
import julius
import librosa
import numpy as np
import soundfile as sf
import omegaconf     
from .utils import(
    sample_fixed_length_data_aligned,
)
from torch import from_numpy
from torch.utils.data import Dataset 

def get_num_segments(files, sample_rate):
    total_num_segement = 0
    for file in files:
        mixture, sr = sf.read(file, dtype="float32")
        total_num_segement += int(mixture.shape[-1]/sr * sample_rate)
    return total_num_segement


class WavDataset(Dataset):
    """
    Define train dataset
    """

    def __init__(self,
                 mixture_dataset,
                 clean_dataset,
                 scenes,
                 sample_length,
                 limit=None,
                 offset=0,
                 normalize="",
                 sample_rate=16000,
                 train=True
                 ):
        """
        Construct train dataset
        Args:
            mixture_dataset (str): mixture dir (wav format files)
            clean_dataset (str): clean dir (wav format files)
            limit (int): the limit of the dataset
            offset (int): the offset of the dataset
        """
        assert os.path.exists(mixture_dataset) and os.path.exists(clean_dataset),  f"Path {mixture_dataset} or {clean_dataset} is not existed..."
        
        print(f"Search dataset from {mixture_dataset}...")

        mixture_wav_files_find = librosa.util.find_files(mixture_dataset, ext="wav", limit=limit, offset=offset)
        clean_wav_files_find = librosa.util.find_files(clean_dataset, ext="wav", limit=limit, offset=offset)
        
        mixture_wav_files_find = sorted(mixture_wav_files_find)
        clean_wav_files_find = sorted(clean_wav_files_find)

        mixture_wav_files_find = [mixture_wav_files_find[i] for i in scenes]
        clean_wav_files_find = [clean_wav_files_find[i] for i in scenes]

        if train:
            index_wav_files = np.arange(len(mixture_wav_files_find))
            random.shuffle(index_wav_files)
            mixture_wav_files = [mixture_wav_files_find[i] for i in index_wav_files]
            clean_wav_files = [clean_wav_files_find[i] for i in index_wav_files]
        if not train:
            mixture_wav_files = mixture_wav_files_find
            clean_wav_files = clean_wav_files_find    

        assert len(mixture_wav_files) == len(clean_wav_files)
        print(f"\t Original length: {len(mixture_wav_files)}")

        self.mixture_wav_files = mixture_wav_files
        self.clean_wav_files = clean_wav_files

        self.train = train
        self.length = len(self.mixture_wav_files)
        self.sample_length = sample_length
        self.normalize = normalize
        self.sample_rate = sample_rate

        print(f"\t Offset: {offset}")
        print(f"\t Limit: {limit}")
        print(f"\t Final length: {self.length}")
        print(f"\t Norm:  {self.normalize}")
            
    def __len__(self):
        return self.length

    def __getitem__(self, item):
        mixture_path = self.mixture_wav_files[item]
        clean_path = self.clean_wav_files[item]
        name = os.path.splitext(os.path.basename(clean_path))[0]
        
        mixture, sr = sf.read(mixture_path, dtype="float32")
        clean, sr = sf.read(clean_path, dtype="float32")
        original_length = mixture.shape[0]

        if len(mixture.shape) == 1:
            mixture = np.expand_dims(mixture, 0)
            clean = np.expand_dims(clean, 0)

        # curr_time = time.perf_counter()

        if sr != self.sample_rate:
            mixture = julius.resample_frac(x=from_numpy(mixture), old_sr=sr, new_sr=self.sample_rate,
                                        output_length=None, full=False)
            clean = julius.resample_frac(x=from_numpy(clean), old_sr=sr, new_sr=self.sample_rate,
                                        output_length=None, full=False)
            sr = self.sample_rate

        # print("\nTime for resample: ", time.perf_counter()-curr_time)
            
        if not self.train:
            return mixture, clean, original_length, name

        if self.train:
            mixture_metadata = {
                "min": 0,
                "max": 0,
                "mean": 0,
                "std": 0,
            }

            clean_metadata = {
                "min": 0,
                "max": 0,
                "mean": 0,
                "std": 0,
            }

            # curr_time = time.perf_counter()

            eps = 1e-6
            if self.normalize == "z-score":
                mixture_metadata["mean"] = torch.mean(mixture, axis=-1, keepdims=True)
                mixture_metadata["std"] = torch.std(mixture, axis=-1, keepdims=True)
                clean_metadata["mean"] = torch.mean(clean, axis=-1, keepdims=True)
                clean_metadata["std"] = torch.std(clean, axis=-1, keepdims=True)
                mixture = (mixture-mixture_metadata["mean"])/(mixture_metadata["std"]+eps)
                clean = (clean-clean_metadata["mean"])/(clean_metadata["std"]+eps)
            
            if self.normalize == "linear-scale":
                mixture_metadata["max"] = torch.max(mixture, axis=-1, keepdims=True)
                mixture_metadata["min"] = torch.min(mixture, axis=-1, keepdims=True)
                clean_metadata["max"] = torch.max(clean, axis=-1, keepdims=True)
                clean_metadata["min"] = torch.min(clean, axis=-1, keepdims=True)
                mixture = (mixture-mixture_metadata["min"])/(mixture_metadata["max"] - mixture_metadata["min"]+eps)
                clean = (clean-clean_metadata["min"])/(clean_metadata["max"] - clean_metadata["min"]+eps)
            
            #     print("\nTime for norm: ", time.perf_counter()-curr_time)
            
            assert sr == self.sample_rate
            assert mixture.shape == clean.shape

            if self.sample_length:
                mixture, clean = sample_fixed_length_data_aligned([mixture, clean], self.sample_length)
            
            return mixture, clean, mixture_metadata, clean_metadata, name       
          
class ClarityWavDataset(Dataset):
    """
    Define train dataset
    """
    def __init__(self,
                 path_dataset,
                 scenes,
                 sample_length,
                 limit=None,
                 offset=0,
                 normalize="",
                 sample_rate=16000,
                 train=True
                 ):
        """
        Construct train dataset
        Args:
            path_dataset (str): dataset dir (wav format files)
            limit (int): the limit of the dataset
            offset (int): the offset of the dataset

            S00001_hr.wav
            S00001_interferer_CH0.wav
            S00001_interferer_CH1.wav
            S00001_interferer_CH2.wav
            S00001_interferer_CH3.wav
            S00001_mix_CH0.wav
            [V] S00001_mix_CH1.wav
            S00001_mix_CH2.wav
            S00001_mix_CH3.wav
            [V] S00001_target_anechoic_CH1.wav
            S00001_target_CH0.wav
            [V] S00001_target_CH1.wav
            S00001_target_CH2.wav
            S00001_target_CH3.wav

            target/
                T001_A08_03103.wav

            interferers/noise/
                CIN_dishwasher_001.wav
            
            interferers/music/00...99
                1009600.low.mp3 
           
            interferers/speech/
                 irm_02484.wav 
            
        """
        assert os.path.exists(path_dataset), f"Path {path_dataset} is not existed..."
        
        source_list = ['hr', 'interferer', 'mix', 'target', 'target_anechoic']
        scene_channel_list = ['CH0', 'CH1', 'CH2', 'CH3']
        self.target_time = omegaconf.OmegaConf.load(os.path.join(path_dataset, "custom_metadata/scenes.train.time.json"))

        # get dataset
        if train:
            mixture_wav_files_find = []
            clean_wav_files_find = []
            interferer_wav_files_find = []
            for scene in scenes:
                for ch in scene_channel_list:
                    clean_wav_file = f'{path_dataset}/train/scenes/{scene}_{source_list[3]}_{ch}.wav'
                    mixture_wav_file = f'{path_dataset}/train/scenes/{scene}_{source_list[2]}_{ch}.wav'
                    interferer_wav_file = f'{path_dataset}/train/scenes/{scene}_{source_list[1]}_{ch}.wav'
                
                    clean_wav_files_find.append(clean_wav_file)
                    mixture_wav_files_find.append(mixture_wav_file)
                    interferer_wav_files_find.append(interferer_wav_file)

            clean_wav_files_find = sorted(clean_wav_files_find)
            mixture_wav_files_find = sorted(mixture_wav_files_find)
            interferer_wav_files_find = sorted(interferer_wav_files_find)

            index_wav_files = np.arange(len(mixture_wav_files_find))
            random.shuffle(index_wav_files)
            
            mixture_wav_files = []
            clean_wav_files = []
            interferer_wav_files = []
            for ifile in index_wav_files:
                mixture_wav_files.append(mixture_wav_files_find[ifile])
                clean_wav_files.append(clean_wav_files_find[ifile])
                interferer_wav_files.append(interferer_wav_files_find[ifile])

            assert len(mixture_wav_files) == len(clean_wav_files) == len(interferer_wav_files)
            
        if not train:
            mixture_wav_files = []
            clean_wav_files = []
            interferer_wav_files = []

            for scene in scenes:
                mixture_wav_file = f'{path_dataset}/train/scenes/{scene}_{source_list[2]}_CH1.wav'
                clean_wav_file = f'{path_dataset}/train/scenes/{scene}_{source_list[3]}_CH1.wav'
                interferer_wav_file = f'{path_dataset}/train/scenes/{scene}_{source_list[1]}_CH1.wav'

                mixture_wav_files.append(mixture_wav_file)
                clean_wav_files.append(clean_wav_file)
                interferer_wav_files.append(interferer_wav_file)
                
        self.mixture_wav_files = mixture_wav_files
        self.clean_wav_files = clean_wav_files
        self.interferer_wav_files = interferer_wav_files

        self.train = train
        self.length = len(self.mixture_wav_files)
        self.sample_length = sample_length
        self.normalize = normalize
        self.sample_rate = sample_rate

        print(f"\t Offset: {offset}")
        print(f"\t Limit: {limit}")
        print(f"\t Final length: {self.length}")
        print(f"\t Norm:  {self.normalize}")
        print(f"\t Sample rate:  {self.sample_rate}")
            
    def __len__(self):
        return self.length

    def __getitem__(self, item):
        mixture_path = self.mixture_wav_files[item]
        clean_path = self.clean_wav_files[item]
        interferer_path = self.interferer_wav_files[item]
        name = os.path.splitext(os.path.basename(clean_path))[0]
        
        mixture, sr = sf.read(mixture_path, dtype="float32")
        clean, sr = sf.read(clean_path, dtype="float32")
        interferer, sr = sf.read(interferer_path, dtype="float32")
        original_length = mixture.shape[0]

        if self.train:
            scene = name.split("_")[0]
            start, end = self.target_time[scene]
            mixture = mixture[start:end, ...]
            clean = clean[start:end, ...]
            interferer = interferer[start:end, ...]

        mixture = np.transpose(mixture, (-1, 0))
        clean = np.transpose(clean, (-1, 0))
        interferer = np.transpose(interferer, (-1, 0))

        if len(mixture.shape) == 1:
            mixture = np.expand_dims(mixture, 0)
            clean = np.expand_dims(clean, 0)
            interferer = np.expand_dims(interferer, 0)
        # else:
        #     mixture = np.mean(mixture, axis=0, keepdims=True)
        #     clean = np.mean(clean, axis=0, keepdims=True)

        # curr_time = time.perf_counter()

        if sr != self.sample_rate:
            mixture = julius.resample_frac(x=from_numpy(mixture), old_sr=sr, new_sr=self.sample_rate,
                                        output_length=None, full=False)
            clean = julius.resample_frac(x=from_numpy(clean), old_sr=sr, new_sr=self.sample_rate,
                                        output_length=None, full=False)
            interferer = julius.resample_frac(x=from_numpy(interferer), old_sr=sr, new_sr=self.sample_rate,
                                        output_length=None, full=False)
            sr = self.sample_rate

        # print("\nTime for resample: ", time.perf_counter()-curr_time)
            
        if not self.train:
            return mixture, clean, interferer, original_length, name

        if self.train:
            mixture_metadata = {
                "min": 0,
                "max": 0,
                "mean": 0,
                "std": 0,
            }

            clean_metadata = {
                "min": 0,
                "max": 0,
                "mean": 0,
                "std": 0,
            }

            interferer_metadata = {
                "min": 0,
                "max": 0,
                "mean": 0,
                "std": 0,
            }

            # curr_time = time.perf_counter()
            
            eps = 1e-6
            if self.normalize == "z-score":
                mixture_metadata["mean"] = torch.mean(mixture, axis=-1, keepdims=True)
                mixture_metadata["std"] = torch.std(mixture, axis=-1, keepdims=True)
                mixture = (mixture-mixture_metadata["mean"])/(mixture_metadata["std"]+eps)
        
                clean_metadata["mean"] = torch.mean(clean, axis=-1, keepdims=True)
                clean_metadata["std"] = torch.std(clean, axis=-1, keepdims=True)
                clean = (clean-clean_metadata["mean"])/(clean_metadata["std"]+eps)

                interferer_metadata["mean"] = torch.mean(interferer, axis=-1, keepdims=True)
                interferer_metadata["std"] = torch.std(interferer, axis=-1, keepdims=True)
                interferer = (interferer-interferer_metadata["mean"])/(interferer_metadata["std"]+eps)
                
            if self.normalize == "linear-scale":
                mixture_metadata["max"] = torch.max(mixture, axis=-1, keepdims=True)
                mixture_metadata["min"] = torch.min(mixture, axis=-1, keepdims=True)
                mixture = (mixture-mixture_metadata["min"])/(mixture_metadata["max"] - mixture_metadata["min"]+eps)
        
                clean_metadata["max"] = torch.max(clean, axis=-1, keepdims=True)
                clean_metadata["min"] = torch.min(clean, axis=-1, keepdims=True)
                clean = (clean-clean_metadata["min"])/(clean_metadata["max"] - clean_metadata["min"]+eps)
                
                interferer_metadata["max"] = torch.max(interferer, axis=-1, keepdims=True)
                interferer_metadata["min"] = torch.min(interferer, axis=-1, keepdims=True)
                interferer = (interferer-interferer_metadata["min"])/(interferer_metadata["max"] - clean_metadata["min"]+eps)
            
            # print("\nTime for norm: ", time.perf_counter()-curr_time)
            
            assert sr == self.sample_rate
            assert mixture.shape == clean.shape

            if self.sample_length:
                mixture, clean, interferer = sample_fixed_length_data_aligned([mixture, clean, interferer], self.sample_length)
            
            return mixture, clean, interferer, mixture_metadata, clean_metadata, interferer_metadata, name

