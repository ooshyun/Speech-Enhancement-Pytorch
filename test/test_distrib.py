import unittest
from torch.utils.data import DataLoader
from src.distrib import (
    get_train_wav_dataset,
    collate_fn_pad,
    get_dataloader,
    get_model,
    get_optimizer,
    get_loss_function,
)
from src.utils import load_yaml

class DistributionSanityCheck(unittest.TestCase):
    def test_load_dataset(self):
        """
        python -m unittest -v test.test_distrib.DistributionSanityCheck.test_load_dataset
        """
        config = load_yaml("./test/conf/config.yaml")
        train_dataset, validation_dataset, test_dataset = get_train_wav_dataset(config=config.dset)

        for d_train in train_dataset:
            mix, sources, mix_metadata, sources_metadata, name = d_train
            break
        print("Train Dataset Length: ", len(train_dataset))

        for d_valid in validation_dataset:
            mix, sources, mix_metadata, sources_metadata, name = d_train
            break
        print("Validation Dataset Length: ", len(validation_dataset))

        for d_test in test_dataset:
            mix, sources, mix_metadata, sources_metadata, name = d_train
            break
        print("Test Dataset Length: ", len(test_dataset))

    def test_load_dataloader(self):
        """
        python -m unittest -v test.test_distrib.DistributionSanityCheck.test_load_dataloader
        """
        config = load_yaml("./test/conf/config.yaml")
        train_dataset, validation_dataset, test_dataset = get_train_wav_dataset(config=config.dset)
        train_dataloader, validation_dataloader = get_dataloader([train_dataset, validation_dataset], config)
        test_dataloader, = get_dataloader([test_dataset, ], config, train=False)

        for d_train in train_dataloader:
            batch_mixture, batch_sources, mixture_metadata_list, sources_metadata_list, names, index_batch = d_train
            print(batch_mixture.size(), batch_sources.size())
            break
        print("Train Dataloader Length: ", len(train_dataloader))
        
        for d_valid in validation_dataloader:
            batch_mixture, batch_sources, mixture_metadata_list, sources_metadata_list, names, index_batch = d_valid
            print(batch_mixture.size(), batch_sources.size())
            break
        print("Validation Dataloader Length: ", len(train_dataloader))

        for d_test in test_dataloader:
            batch_mixture, batch_sources, original_length, name = d_test
            print(batch_mixture.size(), batch_sources.size())
            break
        print("Test Dataloader Length: ", len(train_dataloader))

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

if __name__=="__main__":
    unittest.main()
