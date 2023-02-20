import torch
import unittest
from src.train import main

class ClaritySanityCheck(unittest.TestCase):
    def test_amplify_torch(self):
        """
        python -m unittest -v test.test_clarity_baseline.ClaritySanityCheck.test_amplify_torch
        """
        print()
        import json
        import librosa
        import numpy as np
        import soundfile as sf
        from pathlib import Path
        from omegaconf import OmegaConf
        from scipy.io import wavfile
        from src.utils import load_yaml, obj2dict
        from src.ha.compressor import CompressorTorch
        from src.ha.amplifier import NALRTorch
        from src.audio import amplify_torch
        
        # Should put root path for dataset
        path_config = "./test/conf/config.yaml"
        config = load_yaml(path_config)        
        cfg = OmegaConf.load(config.ha)

        with open(cfg.path.listeners_file, "r", encoding="utf-8") as fp:
            listener_audiograms = json.load(fp)
        
        scenes_folder = Path(cfg.path.scenes_folder)
        
        audiogram = list(listener_audiograms.values())[0]

        # Read signals
        fs_signal, signal = wavfile.read(
            scenes_folder / f"S00001_target_CH1.wav"
        )

        signal = signal / 32768.0

        signal = signal[:4*fs_signal, ...]
        batch = torch.from_numpy(signal.T)
        batch = torch.stack([batch, batch], dim=0) # batch, nchannel, nsamples
        batch = batch.unsqueeze(1) 
        enhancer = NALRTorch(**obj2dict(config.nalr))
        compressor = CompressorTorch(**obj2dict(config.compressor))
        signal_amplfied = amplify_torch(signal=batch, 
                                        enhancer=enhancer,
                                        compressor=compressor,
                                        audiogram=audiogram,
                                        soft_clip=True)
        
        print(signal_amplfied.shape)