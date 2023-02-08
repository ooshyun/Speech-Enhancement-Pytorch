import torch
import unittest
from mllib.src.train import (
    main
)
from recipes.icassp_2023.MLbaseline.evaluate import run_calculate_si
from recipes.icassp_2023.MLbaseline.enhance import enhance

class ClaritySanityCheck(unittest.TestCase):
    def test_dataset(self):
        print()
        """
        python -m unittest -v mllib.test.test_clarity_baseline.ClaritySanityCheck.test_dataset
        """
        from mllib.src.distrib import get_train_wav_dataset
        from mllib.src.utils import load_yaml
        path_config = "./mllib/test/conf/config.yaml"
        config = load_yaml(path_config)
        train_dataset, validation_dataset, test_dataset = get_train_wav_dataset(config.dset, config.default.dset.name)

        for batch_train in train_dataset:
            mixture, clean, mixture_metadata, clean_metadata, name = batch_train
            print(mixture.shape, clean.shape, name)
            break

        for batch_valid in validation_dataset:
            mixture, clean, mixture_metadata, clean_metadata, name = batch_valid
            break

        for batch_test in test_dataset:
            mixture, clean, origial_length, name = batch_test
            print(mixture.shape, clean.shape, name)
            break

    def test_inference(self):
        """
        python -m unittest -v mllib.test.test_clarity_baseline.ClaritySanityCheck.test_inference
        """
        print()
        from mllib.src.evaluate import evaluate
        from mllib.src.distrib import get_train_wav_dataset
        from mllib.src.utils import load_yaml
        path_config = "./mllib/test/conf/config.yaml"
        config = load_yaml(path_config)
        train_dataset, validation_dataset, test_dataset = get_train_wav_dataset(config.dset, config.default.dset.name)

        for batch_valid in test_dataset:
            mixture, clean, origial_length, name = batch_valid
            print("Input: ", mixture.shape)
            mixture = torch.unsqueeze(mixture, dim=0)
            batch, nchannel, nsample = mixture.shape
            mixture = torch.reshape(mixture, shape=(batch*nchannel, 1, nsample))
            enhanced = evaluate(mixture=mixture, model=None, device=torch.device('cpu'), config=config)
            
            assert enhanced.shape ==  mixture.shape
            assert (enhanced-mixture).max() < 1e-6
            
            break

    def test_train(self):
        """
        python -m unittest -v mllib.test.test_clarity_baseline.ClaritySanityCheck.test_train
        """
        print()        
        main("./mllib/test/conf/config.yaml")

    def test_enhance(self):
        """
        python -m unittest -v mllib.test.test_clarity_baseline.ClaritySanityCheck.test_enhance
        """
        # enhance()
        ...

    def test_evaluate(self):
        """
        python -m unittest -v mllib.test.test_clarity_baseline.ClaritySanityCheck.test_evaluate
        """
        # run_calculate_si
        ...