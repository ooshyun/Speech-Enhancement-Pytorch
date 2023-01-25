import random
import torch
from torch.utils.data import DataLoader
import numpy as np
from .utils import load_yaml
from .distrib import (
    get_train_wav_dataset,
    get_loss_function,
    get_model,
    get_optimizer,
    collate_fn_pad,
)

from .solver import Solver

def main(path_config):
    config = load_yaml(path_config)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    train_dataset, validation_dataset = get_train_wav_dataset(config.dset)

    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=config.solver.batch_size,
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
                                batch_size=config.solver.batch_size,
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
                                # pin_memory_de
                                # vice=,
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

    solver.train()
