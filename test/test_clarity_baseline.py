import unittest
from mllib.src.train import (
    main
)
from recipes.icassp_2023.MLbaseline.evaluate import run_calculate_si
from recipes.icassp_2023.MLbaseline.enhance import enhance

class ClaritySanityCheck(unittest.TestCase):
    def test_dataset(self):
        """
        python -m unittest -v mllib.test.test_clarity_baseline.ClaritySanityCheck.test_dataset
        """
        from mllib.src.distrib import get_train_wav_dataset
        from mllib.src.utils import load_yaml
        path_config = "./mllib/src/conf/config.yaml"
        config = load_yaml(path_config)
        train_dataset, validation_dataset = get_train_wav_dataset(config.dset, config.default.dset)

        for batch_train in train_dataset:
            mixture, clean, mixture_metadata, clean_metadata, name = batch_train
            print(mixture.shape, clean.shape, name)
            break

        for batch_valid in validation_dataset:
            mixture, clean, mixture_metadata, clean_metadata, name = batch_valid
            print(mixture.shape, clean.shape, name)
            break

    def test_train(self):
        """
        python -m unittest -v mllib.test.test_clarity_baseline.ClaritySanityCheck.test_train
        """        
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