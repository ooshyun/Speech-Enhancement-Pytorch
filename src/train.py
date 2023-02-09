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

def main(path_config, return_solver=False, mode="train"):
    config = load_yaml(path_config)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    print("-"*30)
    print("\tSearch training datasets...")
    train_dataset, validation_dataset, test_dataset = get_train_wav_dataset(config.dset)
    
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
        print("-"*30)
        print("\tReturning Solver...")    
        return solver
    else:
        print("-"*30)
        print(f"\t{mode} mode Solver...")    
        if mode=="train":
            solver.train()
        elif mode=="validation":
            solver._run_one_epoch(1, 1, train=False)
        elif mode=="test":
            solver.inference(1, 1)
        
        score = solver.score
        score_inference = solver.score_inference
        score_inference_ref = solver.score_inference_reference
        print("-"*30)
        print("\tResult...")
        
        print("-"*30)
        print("\tScore")
        print(score)

        print("-"*30)
        print("\tScore in test dataset")
        print(score_inference)

        print("-"*30)
        print("\tScore in test reference dataset")
        print(score_inference_ref)
