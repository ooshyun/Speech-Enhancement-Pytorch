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

    def test_solver_stft(self):
        """
        python -m unittest -v test.test_train.TrainSanityCheck.test_solver_stft
        """
        import random
        import torch
        from torch.utils.data import DataLoader
        import numpy as np
        from src.utils import load_yaml
        from src.distrib import (
            get_train_wav_dataset,
            get_loss_function,
            get_model,
            get_optimizer,
            collate_fn_pad,
        )

        from src.solver import Solver
        path_config = "./test/conf/config.yaml"
        config = load_yaml(path_config)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        
        train_dataset, validation_dataset = get_train_wav_dataset(config.dset)

        train_dataloader = DataLoader(dataset=train_dataset,
                                    batch_size=config.solver.batch_size,
                                    shuffle=True,
                                    collate_fn=collate_fn_pad(config.dset, drop_last=True),
                                    )

        validation_dataloader = DataLoader(dataset=validation_dataset,
                                    batch_size=config.solver.batch_size,
                                    shuffle=True,
                                    collate_fn=collate_fn_pad(config.dset, drop_last=True), # preprocess
                                    )
                                    
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
        )

