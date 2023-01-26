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
        
        train_dataset, validation_dataset = get_train_wav_dataset(config.dset)
        train_dataloader, validation_dataloader = get_dataloader(train_dataset, config), get_dataloader(validation_dataset, config)

                                    
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

        for batch_train in train_dataloader:
            if len(batch_train) == 4:
                mixture, clean, name, index = batch_train
            else:
                mixture, clean, mixture_metadata, clean_metadata, name, index = batch_train

            n_batch = np.random.randint(low=0, high=mixture.shape[0])

            mixture = mixture[n_batch, ...]
            mixture = torch.unsqueeze(mixture, dim=0)
            clean = clean[n_batch, ...]
            clean = torch.unsqueeze(clean, dim=0)

            print(mixture.shape, clean.shape)
            mixture_stft = solver._stft(mixture)
            clean_stft = solver._stft(clean)
            print(mixture_stft.shape, clean_stft.shape)
            
            mixture_istft = solver._istft(mixture_stft)
            clean_istft = solver._istft(clean_stft)
            print(mixture_istft.shape, clean_istft.shape)
            
            # fig, (ax0, ax1) = plt.subplots(nrows=2)
            # ax0.plot(mixture_istft[0, 0])
            # ax1.plot(mixture[0, 0])
            # plt.show()

            print("[Mixture] Max diff: ", (mixture_istft-mixture).abs().max())
            print("[Clean] Max diff: ", (clean_istft-clean).abs().max())

            assert (mixture_istft-mixture).abs().max() < 1e-5
            
