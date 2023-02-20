import unittest


class EvaluateSanityCheck(unittest.TestCase):
    def test_evaluate(self):
        """
        python -m unittest -v test.test_eval.EvaluateSanityCheck.test_evaluate
        """
        import torch
        import random
        import librosa
        import numpy as np
        from src.utils import load_yaml
        from src.evaluate import evaluate
        path_config = "./test/conf/config.yaml"
        config = load_yaml(path_config)

        error_rate = 1e-10

        wav, sr = librosa.load(librosa.ex("trumpet"), sr=config.dset.sample_rate)

        wav = wav[..., int(config.dset.sample_rate*config.dset.segment)]
        wav = torch.tensor(wav, dtype=torch.float32)
        
        batch = torch.empty(size=(1, 
                                config.model.audio_channels, 
                                int(config.dset.sample_rate*config.dset.segment)))
        
        batch, wav = torch.broadcast_tensors(batch, wav.unsqueeze(0).unsqueeze(0))
        batch = wav

        # Wavform model
        config.model.name = "conv-tasnet"
        result = evaluate(batch, model=None, device=torch.device("cpu"), config=config)
        assert (batch.numpy() - result.numpy() <error_rate).all()

        # STFT model
        config.model.name = 'dnn'
        result = evaluate(batch, model=None, device=torch.device("cpu"), config=config)
        assert (batch.numpy() - result.numpy() <error_rate).all()
