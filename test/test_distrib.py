import unittest
from torch.utils.data import DataLoader
from src.distrib import (
    get_train_wav_dataset,
    collate_fn_pad,
    get_model,
    get_optimizer,
    get_loss_function,
)
from src.utils import load_yaml

class DistributionSanityCheck(unittest.TestCase):
    def test_load_dataset(self):
        """python -m unittest -v test.test_distrib.DistributionSanityCheck.test_load_dataset
        """
        config = load_yaml("./test/conf/config.yaml")
        train_dataset, validation_dataset = get_train_wav_dataset(config=config.dset)

        for d_train in train_dataset:
            print(d_train) # mixture, clean, name
        print("Dataset Length: ", len(train_dataset))

        for d_valid in validation_dataset:
            print(d_valid) # mixture, clean, name
        print("Dataset Length: ", len(validation_dataset))

    def test_load_dataloader(self):
        """python -m unittest -v test.test_distrib.DistributionSanityCheck.test_load_dataloader
        """
        config = load_yaml("./test/conf/config.yaml")
        train_dataset, validation_dataset = get_train_wav_dataset(config=config.dset)
        train_dataloader = DataLoader(dataset=train_dataset,
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    # sampler=,
                                    # batch_sampler=,
                                    # num_workers=,
                                    collate_fn=collate_fn_pad(config.dset, drop_last=True),
                                    # pin_memory=,
                                    # drop_last=,
                                    # timeout=,
                                    # worker_init_fn=,
                                    # multiprocessing_context=,
                                    # generator=,
                                    # prefetch_factor=,
                                    # persistent_workers=,
                                    # pin_memory_device=,
                                    )

        validation_dataloader = DataLoader(dataset=validation_dataset,
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    # sampler=,
                                    # batch_sampler=,
                                    # num_workers=,
                                    collate_fn=collate_fn_pad(config.dset, drop_last=True), # preprocess
                                    # pin_memory=,
                                    # drop_last=,
                                    # timeout=,
                                    # worker_init_fn=,
                                    # multiprocessing_context=,
                                    # generator=,
                                    # prefetch_factor=,
                                    # persistent_workers=,
                                    # pin_memory_device=,
                                    )


        for d_train in train_dataloader:
            mixture, clean, name = d_train
            print(mixture.size(), clean.size())
        print("Dataloader Length: ", len(train_dataloader))
        
        for d_valid in validation_dataloader:
            mixture, clean, name = d_valid
            print(mixture.size(), clean.size())
        print("Dataloader Length: ", len(train_dataloader))

    def test_load_model(self):
        """
        python -m unittest -v test.test_distrib.DistributionSanityCheck.test_load_model
        """
        config = load_yaml("./test/conf/config.yaml")
        model = get_model(config.model)
        model_size = sum(p.numel() for p in model.parameters()) * 4 / 2**20
        print(f"\n{config.model.name} model size: {model_size:.1f}MB")

    def test_load_optimizer(self):
        """
        python -m unittest -v test.test_distrib.DistributionSanityCheck.test_load_optimizer
        """
        config = load_yaml("./test/conf/config.yaml")
        model = get_model(config.model)
        optimizer = get_optimizer(config.optim, model)
        print(optimizer)

    def test_load_loss(self):
        """
        python -m unittest -v test.test_distrib.DistributionSanityCheck.test_load_loss
        """
        config = load_yaml("./test/conf/config.yaml")
        loss_function = get_loss_function(config.optim)
        print(loss_function)

    
    def test_load_metric(self):
        ...
    # Model 
    # model = get_model(config.model)

    # # Optimizer
    # optimizer = get_optimizer(config.optim)

    # # Loss function
    # loss_function = get_loss_function(config.loss)

    # # Metric?        
if __name__=="__main__":
    unittest.main()
