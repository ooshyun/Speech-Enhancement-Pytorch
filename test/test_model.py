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
        
        train_dataset, validation_dataset, test_dataset = get_train_wav_dataset(config.dset)
        train_dataloader, validation_dataloader = get_dataloader([train_dataset, validation_dataset], config)
        test_dataloader, = get_dataloader(datasets=[test_dataset], config=config, train=False)


        model_list = ['dnn',    # O
                     'unet',    # O
                     'mel-rnn', # O
                     'dccrn',  # TODO: Test since GPU is using fully
                     'dcunet', # O
                     'demucs', # O
                     'wav-unet', # O
                     'conv-tasnet', # O, gpu 19421MiB -> decrease size 
                     'crn', # TODO: X, out nan
                     ]
        index_model = -2
        # index_model = 2
        config.model.name = model_list[index_model]
        model = get_model(config.model)
        optimizer = get_optimizer(config.optim, model)
        loss_function = get_loss_function(config.optim)

        print(model)

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
    
    def test_sepformer(self):
        """
        python -m unittest -v test.test_model.ModelSanityCheck.test_sepformer
        """
        import torch
        from src.model.sepformer.sepformer import SepformerSeparation
        from src.utils import dict2obj
        def get_model():
            return SepformerSeparation
            
        device = 'cpu'

        args = {}

        args["sample_rate"] = 16000
        args["segment"] = 1.024 
        args["num_source"] = 2
        args["encoder_kernel_size"] =16
        args["encoder_in_nchannels"] =1
        args["encoder_out_nchannels"] =256
        args["masknet_chunksize"] =250
        args["masknet_numlayers"] =2
        args["masknet_norm"] ="ln"
        args["masknet_useextralinearlayer"] =False,
        args["masknet_extraskipconnection"] =True,
        args["masknet_numspks"] =10
        args["intra_numlayers"] =8
        args["inter_numlayers"] =8
        args["intra_nhead"] =8
        args["inter_nhead"] =8
        args["intra_dffn"] =1024
        args["inter_dffn"] =1024
        args["intra_use_positional"] =True,
        args["inter_use_positional"] =True,
        args["intra_norm_before"] =True,
        args["inter_norm_before"] =True,

        args = dict2obj(args)
        
        model = get_model()(encoder_kernel_size=args.encoder_kernel_size,
                            encoder_in_nchannels=args.encoder_in_nchannels,
                            encoder_out_nchannels=args.encoder_out_nchannels,
                            masknet_chunksize=args.masknet_chunksize,
                            masknet_numlayers=args.masknet_numlayers,
                            masknet_norm=args.masknet_norm,
                            masknet_useextralinearlayer=args.masknet_useextralinearlayer,
                            masknet_extraskipconnection=args.masknet_extraskipconnection,
                            masknet_numspks=args.masknet_numspks,
                            intra_numlayers=args.intra_numlayers,
                            inter_numlayers=args.inter_numlayers,
                            intra_nhead=args.intra_nhead,
                            inter_nhead=args.inter_nhead,
                            intra_dffn=args.intra_dffn,
                            inter_dffn=args.inter_dffn,
                            intra_use_positional=args.intra_use_positional,
                            inter_use_positional=args.inter_use_positional,
                            intra_norm_before=args.intra_norm_before,
                            inter_norm_before=args.inter_norm_before,).to(device)
        
        length = int(args.sample_rate*args.segment) 
        print(f"Input wav length: {length}")
        
        # in_channels = encoder_in_nchannels
        # out_channels = masknet_numspks
        x = torch.randn(length, args.encoder_in_nchannels).to(device)
        if args.encoder_in_nchannels == 1:
            x = torch.squeeze(x, dim=1)

        print(f"in: {x.shape}")
        batch = x[None]
        batch = torch.cat([x[None], x[None]], dim=0)
        out = model.forward(batch)
        print(f"Out: {out.shape}")

        model_size = sum(p.numel() for p in model.parameters()) * 4 / 2**20
        print(f"model size: {model_size:.1f}MB")    