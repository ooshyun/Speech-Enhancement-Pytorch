import random
import torch
import numpy as np
from .utils import load_yaml
from .distrib import (
    get_train_wav_dataset,
    get_dataloader,
    get_loss_function,
    get_model,
    get_optimizer,
)

from .solver import Solver

def main(path_config):
    config = load_yaml(path_config)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    train_dataset, validation_dataset = get_train_wav_dataset(config.dset, config.default.dset)
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

    solver.train()
