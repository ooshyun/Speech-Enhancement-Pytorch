import os
import julius
import librosa
import numpy as np
import soundfile as sf
from .utils import(
    sample_fixed_length_data_aligned
)
from torch import from_numpy
from torch.utils.data import Dataset 

class WavDataset(Dataset):
    """
    Define train dataset
    """

    def __init__(self,
                 mixture_dataset,
                 clean_dataset,
                 sample_length,
                 limit=None,
                 offset=0,
                 ):
        """
        Construct train dataset
        Args:
            mixture_dataset (str): mixture dir (wav format files)
            clean_dataset (str): clean dir (wav format files)
            limit (int): the limit of the dataset
            offset (int): the offset of the dataset
        """
        print(mixture_dataset, clean_dataset)
        assert os.path.exists(mixture_dataset) and os.path.exists(clean_dataset)

        print("Search datasets...")
        mixture_wav_files = librosa.util.find_files(mixture_dataset, ext="wav", limit=limit, offset=offset)
        clean_wav_files = librosa.util.find_files(clean_dataset, ext="wav", limit=limit, offset=offset)

        assert len(mixture_wav_files) == len(clean_wav_files)
        print(f"\t Original length: {len(mixture_wav_files)}")

        self.length = len(mixture_wav_files)
        self.sample_length = sample_length
        self.mixture_wav_files = mixture_wav_files
        self.clean_wav_files = clean_wav_files

        print(f"\t Offset: {offset}")
        print(f"\t Limit: {limit}")
        print(f"\t Final length: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        mixture_path = self.mixture_wav_files[item]
        clean_path = self.clean_wav_files[item]
        name = os.path.splitext(os.path.basename(clean_path))[0]

        mixture, sr = sf.read(mixture_path, dtype="float32")
        clean, sr = sf.read(clean_path, dtype="float32")

        if len(mixture.shape) == 1:
            mixture = np.expand_dims(mixture, 0)
            clean = np.expand_dims(clean, 0)

        if sr != 16000:
            mixture = julius.resample_frac(from_numpy(mixture), sr, 16000)
            clean = julius.resample_frac(from_numpy(clean), sr, 16000)
            sr = 16000

        assert sr == 16000
        assert mixture.shape == clean.shape

        if self.sample_length:
            mixture, clean = sample_fixed_length_data_aligned(mixture, clean, self.sample_length)


        return mixture, clean, name


