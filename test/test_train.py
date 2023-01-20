import unittest
from src.train import (
    main
)

class TrainSanityCheck(unittest.TestCase):
    def test_train(self):
        """
        python -m unittest -v test.test_train.TrainSanityCheck.test_train
        """
        main("./test/conf/config.yaml")