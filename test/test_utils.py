import unittest
from src.train import (
    main
)
from src.utils import pad_last

class UtilitySanityCheck(unittest.TestCase):
    def test_pad_last(self):
        """
        python -m unittest -v test.test_utils.UtilitySanityCheck.test_pad_last
        """
        import torch
        print()
        x = torch.ones(size=(20,))
        x = pad_last(x, 10)

        assert x.shape == (30,)
        assert (x[..., -10:].numpy()==0).all()
        print("\t Pass 1D tensor...")

        x = torch.ones(size=(20, 20))
        x = pad_last(x, 10)

        assert x.shape == (20, 30)
        assert (x[..., -10:].numpy()==0).all()
        print("\t Pass 2D tensor...")
        
        x = torch.ones(size=(20, 20, 20))
        x = pad_last(x, 10)

        assert x.shape == (20, 20, 30)
        assert (x[..., -10:].numpy()==0).all()
        print("\t Pass 3D tensor...")
        
        x = torch.ones(size=(20, 20, 20, 20))
        x = pad_last(x, 10)

        assert x.shape == (20, 20, 20, 30)
        assert (x[..., -10:].numpy()==0).all()
        print("\t Pass 4D tensor...")


        x = torch.ones(size=(20, 20, 20, 20, 20))
        x = pad_last(x, 10)
        assert x.shape == (20, 20, 20, 20, 30)
        assert (x[..., -10:].numpy()==0).all()
        print("\t Pass 5D tensor...")