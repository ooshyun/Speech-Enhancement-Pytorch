import os
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

import warnings
from .solver import Solver

def main(path_config, return_solver=False):
    config = load_yaml(path_config)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    print("-"*30)
    print("\tSearch training datasets...")
    train_dataset, validation_dataset, test_dataset = get_train_wav_dataset(config.dset, config.default.dset.name)
    
    print("-"*30)
    print("\tLoading data loader...")    
    num_worker = os.cpu_count()
    if config.solver.num_workers > num_worker:
        warnings.warn(f"The number of workers is over the range, it sets {config.solver.num_workers} to {num_worker}...")
        config.solver.num_workers = num_worker
    print("\tThe number of CPU: ", num_worker)
    train_dataloader, validation_dataloader = get_dataloader(datasets=[train_dataset, validation_dataset], config=config, train=True)
    test_dataloader, = get_dataloader(datasets=[test_dataset], config=config, train=False)

    print("-"*30)
    print("\tLoading Model...")                                
    model = get_model(config.model)
    print(model)

    print("-"*30)
    print("\tLoading Optimizer...")
    optimizer = get_optimizer(config.optim, model)
    print(optimizer)
    
    print("-"*30)
    print("\tLoading Loss function...")
    loss_function = get_loss_function(config.optim)
    print(loss_function)

    print("-"*30)
    print("\tLoading Solver...")
    solver = Solver(
        config = config,
        model = model,
        loss_function = loss_function,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        test_dataloader=test_dataloader,
    )

    if return_solver:
        return solver
    else:
        solver.train()
