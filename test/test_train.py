import unittest
from src.train import (
    main
)
from src.utils import load_yaml
from src.evaluate import stft_custom, istft_custom

class TrainSanityCheck(unittest.TestCase):
    def test_train(self):
        """
        python -m unittest -v test.test_train.TrainSanityCheck.test_train
        """
        main("./test/conf/config.yaml", device="cpu")

    def test_train_deverb(self):
        """
        python -m unittest -v test.test_train.TrainSanityCheck.test_train_deverb
        """
        print()
        solver = main("./test/conf/config.yaml", device="cpu", return_solver=True)
        config = solver.config
        print(f"Mode Dataset: {config.dset.mode}, Solver: {config.solver.mode}")

        solver.train()


        config.dset.mode = "deverb"
        config.solver.mode = "deverb"
        config.resume = solver.root_dir.as_posix()

        del solver

        print(f"Mode Dataset: {config.dset.mode}, Solver: {config.solver.mode}")
        main(config, device="cpu")


    def test_stft(self):
        """
        python -m unittest -v test.test_train.TrainSanityCheck.test_stft
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
        
        train_dataset, validation_dataset, test_set = get_train_wav_dataset(config.dset)
        train_dataloader, validation_dataloader = get_dataloader([train_dataset, validation_dataset], config)

                                    
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

            mixture, sources, mixture_metadata, sources_metadata, name, index = batch_train

            n_batch = np.random.randint(low=0, high=mixture.shape[0])

            print(mixture.shape, sources.shape)
            mixture_stft = stft_custom(tensor=mixture, config=config)
            sources_stft = stft_custom(tensor=sources, config=config)
            print(mixture_stft.shape, sources_stft.shape)

            mixture_istft = istft_custom(tensor=mixture_stft, config=config)
            sources_istft = istft_custom(tensor=sources_stft, config=config)
            print(mixture_istft.shape, sources_istft.shape)
            
            # fig, (ax0, ax1) = plt.subplots(nrows=2)
            # ax0.plot(mixture_istft[0, 0])
            # ax1.plot(mixture[0, 0])
            # plt.show()

            print("[Mixture] Max diff: ", (mixture_istft-mixture).abs().max())
            print("[Sources] Max diff: ", (sources_istft-sources).abs().max())

            assert (mixture_istft-mixture).abs().max() < 1e-5
            break
            
    def test_inference(self):
        """
        python -m unittest -v test.test_train.TrainSanityCheck.test_inference
        """
        path_config = "./test/conf/config.yaml"
        solver = main(path_config=path_config, return_solver=True)
        solver.config.solver.test.total_steps = 10
        solver.inference(epoch=1, total_epoch=1)
        score = solver.score_inference
        print(score)