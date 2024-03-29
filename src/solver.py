"""
Most of parts of this codes are from https://github.com/haoxiangsnr/Wave-U-Net-for-Speech-Enhancement.git

Model list
----------
    'mel-rnn': MelRNN,
    'dccrn': DCCRN,
    'dcunet': DCUnet,
    'demucs': Demucs,
    'wav-unet': WavUnet,
    'conv-tasnet': ConvTasNet,
    'crn': CRN,
    'dnn': DeepNeuralNetwork,

WAV mode
----------
    - dccrn
    - wav-unet
    - conv-tasnet
    - demucs

STFT model(real, imag)
----------
    - mel-rnn
    - dcunet
    - crn

STFT model(amplitude)
----------
    - crn
    - dnn
    - rnn types(mel-rnn)

"""
import os
import time
import json
import tqdm
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt

from pathlib import Path
from shutil import copyfile

import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.autograd

from torch.utils.tensorboard import SummaryWriter
from .evaluate import (
    evaluate,
    stft_custom,
    istft_custom,
)
# import torch.autograd.profiler as prof

from .loss import UtterenceBaasedPermutationInvariantTraining

from .metric import (
    WB_PESQ,
    STOI,
    SI_SDR,
    SpeechMetricResultsFile,
)

from .utils import (
    prepare_device,
    obj2dict,
    load_yaml,
    get_filtered_snr_file,
)

from .audio import amplify_torch
from .ha.amplifier import NALRTorch
from .ha.compressor import CompressorTorch

try:
    import julius
    from omegaconf import OmegaConf
    from recipes.icassp_2023.MLbaseline.metrics import evaluate_clarity
    LIB_CLARITY = True
except ModuleNotFoundError:
    print("There's no clarity library")
    LIB_CLARITY = False

try:
    from torchmetrics import ScaleInvariantSignalDistortionRatio
    from torchmetrics.audio import (
    ShortTimeObjectiveIntelligibility,
    PerceptualEvaluationSpeechQuality,
    PIT,
)
    LIB_TORCH_METRIC = True
except ImportError:
    LIB_TORCH_METRIC = False

from .model.types import (MULTI_SPEECH_SEPERATION_MODELS,
                MULTI_CHANNEL_SEPERATION_MODELS,
                MONARCH_SPEECH_SEPARTAION_MODELS, 
                STFT_MODELS,
                WAV_MODELS,
)

class Solver(object):
    def __init__(self, 
                config, 
                model, 
                optimizer=None, 
                loss_function=None,
                train_dataloader=None,
                validation_dataloader=None,
                test_dataloader=None,
                device='gpu',
                ):
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        
        if config.ha:
            ha_cfg = OmegaConf.load(config.ha)
            with open(ha_cfg.path.listeners_file, "r", encoding="utf-8") as fp:
                listener_audiograms = json.load(fp)                                    
            self.audiogram_dataloader = list(listener_audiograms.values()) # num = 83
        else:
            self.audiogram_dataloader = None

        self.n_gpu = torch.cuda.device_count()
        self.device = prepare_device(self.n_gpu, cudnn_deterministic=config.solver.cudnn_deterministic) if device == 'gpu' else torch.device('cpu')
        print("-"*30)
        print(f"\tLoading training in {self.device}...")  

        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

        self.optimizer = optimizer
        self.loss_function = loss_function

        self.model = model.to(self.device)

        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(self.n_gpu)))

        # Trainer
        self.epochs = config.solver.epochs
        self.save_checkpoint_interval = config.solver.save_checkpoint_interval
        self.validation_interval = config.solver.validation.interval
        self.test_interval = config.solver.test.interval
        self.num_visualization = 0

        # The following args is not in the config file, We will update it if resume is True in later.
        self.find_max = True if config.solver.validation.metric in ("stoi", "pesq", "sisdr", "haspi", "hasqi") else False
        self.score = {"best_score": -np.inf if self.find_max else np.inf,
                    "loss": 0,
                    "loss_valid": 0,
                    "grad_norm": 0,
                    "stoi": [],
                    "pesq": [],
                    "sisdr": [],
                    "haspi": [],
                    "hasqi": [],}

        self.score_inference = {"loss": 0,
                                "stoi": [],
                                "pesq": [],
                                "sisdr": [],
                                "haspi": [],
                                "hasqi": [],}

        self.score_inference_reference = {"loss": 0,
                                        "stoi": [],
                                        "pesq": [],
                                        "sisdr": [],
                                        "haspi": [],
                                        "hasqi": [],}

        if LIB_TORCH_METRIC:
            self.metric_torch_reference = {"stoi": ShortTimeObjectiveIntelligibility(fs=config.model.sample_rate, extended=True),
                        "pesq": PerceptualEvaluationSpeechQuality(fs=config.model.sample_rate, mode = "wb"),
                        "sisdr": ScaleInvariantSignalDistortionRatio(zero_mean=False), }
            
            self.metric_torch_estimation = {"stoi": ShortTimeObjectiveIntelligibility(fs=config.model.sample_rate, extended=True),
                        "pesq": PerceptualEvaluationSpeechQuality(fs=config.model.sample_rate, mode = "wb"),
                        "sisdr": ScaleInvariantSignalDistortionRatio(zero_mean=False), }
            

        self.metric = {"stoi": STOI,
                    "pesq": WB_PESQ,
                    "sisdr": SI_SDR}

        self.root_dir = Path(config.solver.root) / "result" / config.model.name / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.checkpoints_dir = self.root_dir / "checkpoints"
        self.logs_dir = self.root_dir / "logs"

        for dir in [self.checkpoints_dir, self.logs_dir]:
            if not dir.exists():
                dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.logs_dir.as_posix(), max_queue=5, flush_secs=30)
        self.writer.add_text(
            tag="Configuration",
            text_string=f"<pre>  \n{json.dumps(obj2dict(config), indent=4, sort_keys=False)}  \n</pre>",
            global_step=1
        )
        self.result_writer = None
        # self.gpu_profiler = prof.profile(enabled=True, 
        #                             use_cuda=True,
        #                             profile_memory=True,
        #                             )
        
        self.config = config
        if config.solver.preloaded_model: 
            self._preload_model()
        else:
            if config.solver.resume: 
                self._resume_checkpoint()

        print(f"\n Save to {self.root_dir}")
        print("\nConfigurations are as follows: ")
        print(obj2dict(config), "\n")        
        copyfile(config.root, (self.root_dir / "config.yaml").as_posix())
        
        print(f"\t Speech Enhancment Model")
        self._print_networks([self.model])

        ### SNR FILES ###
        self.file_name_list = None
        # self.file_name_list = get_filtered_snr_file(config)

    def _resume_checkpoint(self):
        """
        Resume experiment from latest checkpoint.
        
        Notes
        ------
        To be careful at Loading model. if model is an instance of DataParallel, we need to set model.module.*
        """
        cpt_path =  "checkpoints/latest_model.tar"
        latest_model_path = Path(self.config.solver.resume) / cpt_path
        assert latest_model_path.exists(), f"{latest_model_path} does not exist, can not load latest checkpoint."

        checkpoint = torch.load(latest_model_path.as_posix(), map_location=self.device)

        # self.start_epoch = checkpoint["epoch"] + 1
        self.best_score = checkpoint["best_score"]
        if self.config.optim.load: 
            print("-"*30)
            print(f"\tOptimizer Loading...")
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint["model"]) 
        else:
            self.model.load_state_dict(checkpoint["model"])

        print("-"*30)
        print(f"\tModel checkpoint loaded, {latest_model_path.as_posix()}")       

    def _preload_model(self):
        """
        Preload *.pth file of the model at the start of the current experiment.

        Args:
            model_path(Path): the path of the *.pth file
        """
        model_path = Path(self.config.solver.preloaded_model)
        assert model_path.exists(), f"Preloaded *.pth file is not exist. Please check the file path: {model_path.as_posix()}"
        model_checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(model_checkpoint, strict=False)
        else:
            self.model.load_state_dict(model_checkpoint, strict=False)

        print("-"*30)
        print(f"\tModel preloaded successfully from {model_path.as_posix()}.")

    @staticmethod
    def _print_networks(nets: list):
        print(f"This project contains {len(nets)} networks, the number of the parameters: ")
        params_of_all_networks = 0
        for i, net in enumerate(nets, start=1):
            params_of_network = 0
            for param in net.parameters():
                params_of_network += param.numel()

            print(f"\tNetwork {i}: {params_of_network / 1e6} million.")
            params_of_all_networks += params_of_network

        print(f"The amount of parameters in the project is {params_of_all_networks / 1e6} million.")

    def _save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint to <root_dir>/checkpoints directory, which contains:
            - current epoch
            - best score in history
            - optimizer parameters
            - model parameters
        Args:
            is_best(bool): if current checkpoint got the best score, it also will be saved in <root_dir>/checkpoints/best_model.tar.
        """
        print(f"\t Saving {epoch} epoch model checkpoint...")

        # Construct checkpoint tar package
        state_dict = {
            "epoch": epoch,
            "best_score": self.score["best_score"],
            "optimizer": self.optimizer.state_dict()
        }

        if isinstance(self.model, torch.nn.DataParallel):  # Parallel
            state_dict["model"] = self.model.module.cpu().state_dict() 
        else:
            state_dict["model"] = self.model.cpu().state_dict()
        """
        Notes:
            - latest_model.tar:
                Contains all checkpoint information, including optimizer parameters, model parameters, etc. New checkpoint will overwrite old one.
            - model_<epoch>.pth: 
                The parameters of the model. Follow-up we can specify epoch to inference.
            - best_model.tar:
                Like latest_model, but only saved when <is_best> is True.
            - state.json:
                score when train
        """

        torch.save(state_dict, (self.checkpoints_dir / "latest_model.tar").as_posix())
        torch.save(state_dict["model"], (self.checkpoints_dir / f"model_{str(epoch).zfill(4)}_{self.config.solver.validation.metric}_{self.score['best_score']:2.8f}.pth").as_posix())
        
        with open(self.checkpoints_dir / "state.json", 'w') as tmp:
            json.dump(self.score, tmp, indent=4) 

        if is_best:
            print(f"\t Found best score in {epoch} epoch, saving...")
            torch.save(state_dict, (self.checkpoints_dir / "best_model.tar").as_posix())

        # Use model.cpu() or model.to("cpu") will migrate the model to CPU, at which point we need re-migrate model back.
        # No matter tensor.cuda() or tensor.to("cuda"), if tensor in CPU, the tensor will not be migrated to GPU, but the model will.
        self.model.to(self.device)

    def _is_best(self, score, find_max=True):
        """Check if the current model is the best model
        """
        if find_max and score >= self.score["best_score"]:
            self.score["best_score"] = score
            return True
        elif not find_max and score <= self.score["best_score"]:
            self.score["best_score"] = score
            return True
        else:
            return False

    def train(self):
        patience = self.config.solver.patience
        early_stopping = 0

        for epoch in range(self.epochs):
            print(f"============== {epoch} / {self.epochs} epoch ==============")
            print("[0 seconds] Begin training...")
            
            start_time = time.time()

            score = self._run_one_epoch(epoch, self.epochs, train=True)
                        
            if epoch % self.save_checkpoint_interval == 0:
                self._save_checkpoint(epoch)

            if epoch % self.validation_interval == 0:
                print(f"[{int(time.time() - start_time)} seconds] Training is over, Validation is in progress...")
                score = self._run_one_epoch(epoch, self.epochs, train=False)
                if self._is_best(score, find_max=self.find_max):
                    self._save_checkpoint(epoch, is_best=True)
                    early_stopping = 0
                else:
                    early_stopping += 1
            
            # if epoch % self.test_interval == 0: # and epoch > 0:
            #     print(f"[{int(time.time() - start_time)} seconds] Training and Validation are over, Test is in progress...")
            #     score = self.inference(epoch, self.epochs)            

            if early_stopping > patience:
                break

            print(f"[{int(time.time() - start_time)} seconds] End this epoch.")

    def _run_one_epoch(self, epoch, total_epoch, train=False):
        """
        WAV mode
        ----------
            - dccrn
            - wav-unet
            - conv-tasnet
            - demucs

        STFT model(real, imag)
        ----------
            - mel-rnn (channel, nfeature, nframe, 2)
            - dcunet (channel, nfeature, nframe, 2)
            - crn, (channel, nfeature, nframe, 2)
        """
        name_dataset = self.config.dset.name
        loss_total = 0.
        grad_norm_total = 0.
        dataloader = self.train_dataloader if train else self.validation_dataloader

        total_step = len(dataloader)
        if (not self.config.solver.all_steps):
            if train and total_step > self.config.solver.total_steps:
                print(f"\tTotal step({self.config.solver.total_steps}) is less then the length of dataset({total_step})...")
                print(f"\tTrain will stop at step {self.config.solver.total_steps}!")
                total_step = self.config.solver.total_steps
            elif not train and total_step > self.config.solver.validation.total_steps:
                print(f"\tTotal step({self.config.solver.validation.total_steps}) is less then the length of dataset({total_step})...")
                print(f"\tValidation will stop at step {self.config.solver.validation.total_steps}!")
                total_step = self.config.solver.validation.total_steps
    
        if self.config.optim.clip_grad:
            print(f"\tClipping gradient maximum {self.config.optim.clip_grad}...")

        tepoch = tqdm.tqdm(dataloader, ncols=120) # [TODO] Search Pytorch progress bar 
        for step, batch in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch+1}")
            
            if step >= total_step:
                break

            mixture, sources, mixture_metadata, sources_metadata, name, index = batch

            mixture = mixture.to(self.device)
            sources = sources.to(self.device)
            
            batch, nchannel, nsample = mixture.shape
            num_spk = sources.shape[1]

            # print("DEBUG: ", mixture.shape, sources.shape,)

            # mono channel to stereo for source separation models
            assert self.config.model.audio_channels == nchannel, f"Channel between {self.config.dset.name} and {self.config.model.name} did not match..."
            assert self.config.model.num_spk == num_spk, f"number of speakers between {self.config.dset.name} and {self.config.model.name} did not match..."
            
            if self.config.model.name in MULTI_SPEECH_SEPERATION_MODELS:
                assert num_spk == len(self.config.model.sources), f"number of speakers between {self.config.dset.name} and {self.config.model.name} did not match..."
                
            if self.config.model.name in MONARCH_SPEECH_SEPARTAION_MODELS: # squeeze dim sources
                    sources = torch.squeeze(sources, dim=1)                

            # if not source separation models, merge batch and channels
            if self.config.model.name in MONARCH_SPEECH_SEPARTAION_MODELS:
                mixture = torch.reshape(mixture, shape=(batch*nchannel, 1, nsample))
                sources = torch.reshape(sources, shape=(batch*num_spk*nchannel, 1, nsample))

            if self.config.model.name in STFT_MODELS:
                # Reference. https://espnet.github.io/espnet/_modules/espnet2/layers/stft.html
                # return shape: [batch, channel, nfeature, nframe, ndtype]
                mixture = stft_custom(tensor=mixture, config=self.config.model)
                sources = stft_custom(tensor=sources, config=self.config.model)

            # with torch.autograd.detect_anomaly(): # figuring out nan grads
            if train: 
                self.model.train()
            else:
                self.model.eval()

            enhanced: torch.Tensor = self.model(mixture)

            # source separation models give out.shape = batch, sources, channels, features
            if self.config.optim.pit and num_spk >= 2:
                loss = UtterenceBaasedPermutationInvariantTraining(enhance=enhanced, 
                                                                target=sources,
                                                                mixture=mixture if self.config.optim.loss == 'psa' else None,
                                                                loss_function=self.loss_function,
                                                                return_comb=False
                                                                )
            else:
                if num_spk >=2 and self.config.optim.loss == 'psa': 
                    mixture, _= torch.broadcast_tensors(mixture.unsqueeze(dim=1), sources)
            
            loss: torch.Tensor = self.loss_function(enhanced, sources) if not self.config.optim.loss == 'psa' else self.loss_function(enhanced, sources, mixture)

            if train:
                self.optimizer.zero_grad()
                
                loss.backward()
                
                if self.config.optim.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.optim.clip_grad)

                self.optimizer.step()

                grad_norm = 0
                for n, p in self.model.named_parameters():
                    grad_norm += (p.grad.sum() ** 2)
                grad_norm = grad_norm.sqrt().item()
                grad_norm_total += grad_norm

            loss = loss.detach().item()
            loss_total += loss
            
            if not train:
                self.writer.add_scalar(f"Validation/Loss_step", loss, epoch*total_step+step)
                tepoch.set_postfix(loss_valid=loss)
            if train:
                self.writer.add_scalar(f"Train/Loss_step", loss, epoch*total_step+step)
                self.writer.add_scalar(f"Train/grad_norm_step", grad_norm, epoch*total_step+step)
                tepoch.set_postfix(loss_train=loss)

        if train:
            self.score["loss"] = loss_total / total_step        
            self.score["grad_norm"] = grad_norm_total / total_step        
            self.writer.add_scalar(f"Train/Loss", self.score["loss"], epoch)
            self.writer.add_scalar(f"Train/Grad_norm", self.score["grad_norm"], epoch)
        
        if not train:
            self.score["loss_valid"] = loss_total / total_step        
            self.writer.add_scalar(f"Validation/Loss", self.score["loss_valid"], epoch)

            # for metric in list(self.metric.keys()):
            #     self.writer.add_scalars(f"Validation/{metric}", {
            #         "clean and noisy": np.mean(self.score_reference[metric]),
            #         "clean and enhanced": np.mean(self.score[metric]),
            #     }, epoch)

            #     self.writer.add_scalars(f"Validation/{metric}_torch", {
            #         "clean and noisy": self.metric_torch_reference[metric].compute(),
            #         "clean and enhanced": self.metric_torch_estimation[metric].compute(),
            #     }, epoch)

        return self.score["loss"] if train else self.score[self.config.solver.validation.metric]

    def inference(self, epoch, total_epoch, save=False):
        name_dataset = self.config.dset.name
        loss_total = 0.
        dataloader = self.test_dataloader

        total_step = len(dataloader)
        if (not self.config.solver.all_steps) and total_step > self.config.solver.test.total_steps:
            print(f"\tTotal step({self.config.solver.test.total_steps}) is less then the length of dataset({total_step})...")
            print(f"\tTrain will stop at step {self.config.solver.total_steps}!")
            total_step = self.config.solver.test.total_steps

        tepoch = tqdm.tqdm(dataloader, ncols=120)
        for step, batch in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            
            if step >= total_step:
                break

            mixture, sources, original_length, name = batch

            # filter based on SNR 
            if self.file_name_list:                     
                if name[0] not in self.file_name_list:
                    continue
            nbatch, nchannel, nsample = mixture.shape
            num_spk = sources.shape[1]

            # mono channel to stereo for source separation models
            assert self.config.model.audio_channels == nchannel, f"Channel between {self.config.dset.name} and {self.config.model.name} did not match..."
            assert self.config.model.num_spk == num_spk, f"number of speakers between {self.config.dset.name} and {self.config.model.name} did not match..."

            if self.config.model.name in MULTI_SPEECH_SEPERATION_MODELS:
                assert num_spk == len(self.config.model.sources), f"number of speakers between {self.config.dset.name} and {self.config.model.name} did not match..."

            # if not source separation models, merge batch and channels
            if self.config.model.name in MONARCH_SPEECH_SEPARTAION_MODELS:
                mixture = torch.reshape(mixture, shape=(nbatch*nchannel, 1, nsample))
            
            enhanced = evaluate(mixture=mixture, model=self.model, device=self.device, config=self.config) 
            
            if self.config.model.name in MONARCH_SPEECH_SEPARTAION_MODELS:
                mixture = torch.reshape(mixture, shape=(nbatch, nchannel, nsample))
                enhanced = torch.reshape(enhanced, shape=(nbatch, nchannel, nsample))

            if self.config.model.name in MONARCH_SPEECH_SEPARTAION_MODELS:
                sources = torch.squeeze(sources, dim=1)
            elif self.config.model.name in MULTI_SPEECH_SEPERATION_MODELS:
                enhanced = enhanced[:, 0, ...]
                sources = sources[:, 0, ...]

            loss: torch.Tensor = self.loss_function(sources, enhanced) if not self.config.optim.loss == 'psa' else self.loss_function(enhanced, sources, mixture)
            loss = loss.detach().item()
            loss_total += loss

            assert sources.shape == enhanced.shape == mixture.shape

            mixture = mixture.detach().cpu()
            sources = sources.detach().cpu()
            enhanced = enhanced.detach().cpu()

            # compute metric
            score, score_reference = self.compute_metric(mixture=mixture, enhanced=enhanced, clean=sources)

            # record metric to tensorboard
            for metric_name, metric_value in score.items():
                self.score_inference[metric_name] += metric_value    
                self.writer.add_scalar(f"Test/{metric_name}_enhance", np.mean(metric_value), (epoch+1)*total_step+step)
            
            for metric_name, metric_value in score_reference.items():
                self.score_inference_reference[metric_name] += metric_value  
                self.writer.add_scalar(f"Test/{metric_name}_mixture", np.mean(metric_value), (epoch+1)*total_step+step)
                
            # record metric to visualization
            self.spec_audio_visualization(mixture=mixture, enhanced=enhanced, clean=sources, name=name[0], epoch=epoch)            
            self.num_visualization = 0

            if LIB_CLARITY and self.config.dset.name == 'Clarity':
                self.compute_metric_clarity(mixture=mixture, enhanced=enhanced, length=original_length, name=name[0])
                tepoch.set_postfix(loss=loss, metric=np.mean(self.score_inference[self.config.solver.test.metric]), metric2=np.mean(self.score_inference["haspi"]))
                self.writer.add_scalar(f"Test/haspi_enhance", np.mean(self.score_inference["haspi"]), (epoch+1)*total_step+step)
                self.writer.add_scalar(f"Test/hasqi_enhance", np.mean(self.score_inference["hasqi"]), (epoch+1)*total_step+step)
                self.writer.add_scalar(f"Test/haspi_mixture", np.mean(self.score_inference_reference["haspi"]), (epoch+1)*total_step+step)
                self.writer.add_scalar(f"Test/hasqi_mixture", np.mean(self.score_inference_reference["hasqi"]), (epoch+1)*total_step+step)
            else:
                tepoch.set_postfix(loss=loss, metric=np.mean(self.score_inference[self.config.solver.test.metric]))

            # (optional) save metric to csv
            if save:
                if self.result_writer is None:
                    self.result_writer = SpeechMetricResultsFile(file_name=self.root_dir/ f"scores_{self.config.model.name}_{self.root_dir.as_posix().split('/')[-1]}.csv")
                    self.result_writer.write_header()

                result_dict = {}
                for metric, score in self.score_inference.items():
                    if isinstance(score, list):
                        result_dict[f"{metric}_enhance"] = score[-1] if len(score) >= 1 else 0
                for metric, score in self.score_inference_reference.items():
                    if isinstance(score, list):
                        result_dict[f"{metric}_mixture"] = score[-1] if len(score) >= 1 else 0
                self.result_writer.add_result(scene=name[0].split("_")[0], **result_dict)

        self.score_inference["loss"] = loss_total / len(dataloader)  
        
        for metric in list(self.score_inference_reference.keys()):
            self.writer.add_scalars(f"Test/{metric}", {
                "clean and noisy": np.mean(self.score_inference[metric]),
                "clean and enhanced": np.mean(self.score_inference_reference[metric]),
            }, epoch)

    def spec_audio_visualization(self, mixture, enhanced, clean, name, epoch):
        # Visualize audio
        if self.num_visualization > self.config.solver.test.num_show:
            return

        # view cause error
        # RuntimeError: view size is not compatible with input tensor's size and stride 
        # (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        # This error cannot solve using .contingous() -> What is differnece between reshape, contingous, view?
        batch, nchannel, nsamples = mixture.size()
        mixture_one_sequence = mixture.reshape(nchannel, batch, nsamples) 
        enhanced_one_sequence = enhanced.reshape(nchannel, batch, nsamples) 
        clean_one_sequence = clean.reshape(nchannel, batch, nsamples) 

        mixture_one_sequence = mixture_one_sequence.reshape(nchannel*batch*nsamples) 
        enhanced_one_sequence = enhanced_one_sequence.reshape(nchannel*batch*nsamples) 
        clean_one_sequence = clean_one_sequence.reshape(nchannel*batch*nsamples) 
    
        # self.writer.add_audio(f"Speech/{name}_mixture", mixture_one_sequence, epoch, sample_rate=self.config.dset.sample_rate)
        # self.writer.add_audio(f"Speech/{name}_Enhanced", enhanced_one_sequence, epoch, sample_rate=self.config.dset.sample_rate)
        # self.writer.add_audio(f"Speech/{name}_Clean", clean_one_sequence, epoch, sample_rate=self.config.dset.sample_rate)
        
        mixture_one_sequence = mixture_one_sequence.numpy()
        enhanced_one_sequence = enhanced_one_sequence.numpy()
        clean_one_sequence = clean_one_sequence.numpy()

        # Visualize waveform
        fig, ax = plt.subplots(3, 1)
        for j, y in enumerate([mixture_one_sequence, enhanced_one_sequence, clean_one_sequence]):
            ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                np.mean(y),
                np.std(y),
                np.max(y),
                np.min(y)
            ))
            librosa.display.waveshow(y, sr=self.config.dset.sample_rate, ax=ax[j])
        plt.tight_layout()
        self.writer.add_figure(f"Waveform/{name}", fig, epoch)

        # Visualize spectrogram
        mixture_mag, _ = librosa.magphase(librosa.stft(mixture_one_sequence, n_fft=320, hop_length=160, win_length=320))
        enhanced_mag, _ = librosa.magphase(librosa.stft(enhanced_one_sequence, n_fft=320, hop_length=160, win_length=320))
        clean_mag, _ = librosa.magphase(librosa.stft(clean_one_sequence, n_fft=320, hop_length=160, win_length=320))

        fig, axes = plt.subplots(3, 1, figsize=(6, 6))
        for k, mag in enumerate([
            mixture_mag,
            enhanced_mag,
            clean_mag,
        ]):
            axes[k].set_title(f"mean: {np.mean(mag):.3f}, "
                            f"std: {np.std(mag):.3f}, "
                            f"max: {np.max(mag):.3f}, "
                            f"min: {np.min(mag):.3f}")
            librosa.display.specshow(librosa.amplitude_to_db(mag), cmap="magma", y_axis="linear", ax=axes[k], sr=16000)
        plt.tight_layout()
        self.writer.add_figure(f"Spectrogram/{name}", fig, epoch)
        self.num_visualization += 1

        # del fig, axes

    def compute_metric(self, mixture, enhanced, clean):
        metric_name = list(self.metric.keys())

        score = {name: [] for name in metric_name}
        score_reference = {name: [] for name in metric_name}

        for metric in metric_name:
            score_mixture = self.metric[metric](estimation=mixture, reference=clean)
            score_enhanced = self.metric[metric](estimation=enhanced, reference=clean)

            score_reference[metric].append(score_mixture)
            score[metric].append(score_enhanced)
            
            if LIB_TORCH_METRIC:
                self.metric_torch_reference[metric].update(preds=mixture, target=clean)
                self.metric_torch_estimation[metric].update(preds=enhanced, target=clean)

        return score, score_reference

    def compute_metric_clarity(self, mixture, enhanced, length, name):
        assert mixture.shape[0] == 1, "Clarity batch can only 1..."
    
        cfg = OmegaConf.load(self.config.ha)
        scene = name.split("_")[0]

        enhanced_resample = enhanced.detach()
        mixture_resample = mixture.detach()
        if self.config.dset.sample_rate != cfg.nalr.fs:
            enhanced_resample = julius.resample_frac(x=enhanced_resample, old_sr=self.config.dset.sample_rate, new_sr=cfg.nalr.fs,
                                                    output_length=None, full=True)
            mixture_resample = julius.resample_frac(x=mixture_resample, old_sr=self.config.dset.sample_rate, new_sr=cfg.nalr.fs,
                                                    output_length=None, full=True)
            
        enhanced_resample = torch.squeeze(enhanced_resample, dim=0).numpy()
        mixture_resample = torch.squeeze(mixture_resample, dim=0).numpy()

        score = evaluate_clarity(scene=scene, enhanced=enhanced_resample, sample_rate=cfg.nalr.fs, cfg=cfg)[0]
        score_mixture = evaluate_clarity(scene=scene, enhanced=mixture_resample, sample_rate=cfg.nalr.fs, cfg=cfg)[0]
        
        self.score_inference['haspi'].append(score[0])
        self.score_inference['hasqi'].append(score[1])
        self.score_inference_reference['haspi'].append(score_mixture[0])
        self.score_inference_reference['hasqi'].append(score_mixture[1])
        