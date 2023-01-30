import unittest

class ModelSanityCheck(unittest.TestCase):
    def test_model(self):
        """
        python -m unittest -v test.test_model.ModelSanityCheck.test_model
        """
        import random
        import torch
        import numpy as np
        from src.utils import load_yaml
        from src.distrib import (
            get_train_wav_dataset,
            get_loss_function,
            get_model,
            get_optimizer,
            get_dataloader,
        )
        import matplotlib.pyplot as plt
        from src.solver import Solver
        path_config = "./test/conf/config.yaml"
        config = load_yaml(path_config)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        
        train_dataset, validation_dataset, test_dataset = get_train_wav_dataset(config.dset, config.default.dset.name)
        train_dataloader, validation_dataloader = get_dataloader([train_dataset, validation_dataset], config)
        test_dataloader, = get_dataloader(datasets=[test_dataset], config=config, train=False)


        model_list = ['mel-rnn', # X
                     'dccrn',  # X, sample in, out differenet, ex. 16324 -> 16300
                     'dcunet', # O
                     'demucs', # O
                     'wav-unet', # O
                     'conv-tasnet', # O, gpu 19421MiB
                     'crn', # X, out nan
                     ]
        index_model = 5

        config.model.name = model_list[index_model]
                     
        model = get_model(config.model)
        optimizer = get_optimizer(config.optim, model)
        loss_function = get_loss_function(config.optim)

        solver = Solver(
            config = config,
            model = model,
            loss_function = loss_function,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            test_dataloader=test_dataloader,
        )
        solver.train()
