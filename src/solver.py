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
    'dnn':

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
import librosa
import soundfile as sf
import librosa.display
import matplotlib.pyplot as plt

from pathlib import Path
from shutil import copyfile

import datetime
import numpy as np

import torch
import torch.autograd

from torch.utils.tensorboard import SummaryWriter
from .evaluate import evaluate

from .loss import PermutationInvariantTraining

from .metric import (
    WB_PESQ,
    STOI,
    SI_SDR,
)

from .utils import (
    prepare_device,
    obj2dict,
    load_yaml
)

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
)
    LIB_TORCH_METRIC = True
except ImportError:
    LIB_TORCH_METRIC = False

class Solver(object):
    def __init__(self, 
                config, 
                model, 
                optimizer=None, 
                loss_function=None,
                train_dataloader=None,
                validation_dataloader=None,
                test_dataloader=None,
                ):
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.n_gpu = torch.cuda.device_count()
        self.device = prepare_device(self.n_gpu, cudnn_deterministic=config.solver.cudnn_deterministic)

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
        self.score = {"best_score": -np.inf,
                    "loss": 0,
                    "loss_valid": 0,
                    "grad_norm": 0,
                    "grad_norm_valid": 0,
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

        self.config = config
        if config.solver.resume: self._resume_checkpoint()
        if config.solver.preloaded_model: self._preload_model() 
        
        print(f"\n Save to {self.root_dir}")
        print("\nConfigurations are as follows: ")
        print(obj2dict(config), "\n")        
        copyfile(config.root, (self.root_dir / "config.yaml").as_posix())
        
        self._print_networks([self.model])

    def _resume_checkpoint(self):
        """
        Resume experiment from latest checkpoint.
        
        Notes
        ------
        To be careful at Loading model. if model is an instance of DataParallel, we need to set model.module.*
        """
        latest_model_path = Path(self.config.solver.resume) / "checkpoints/latest_model.tar"
        assert latest_model_path.exists(), f"{latest_model_path} does not exist, can not load latest checkpoint."

        checkpoint = torch.load(latest_model_path.as_posix(), map_location=self.device)

        # self.start_epoch = checkpoint["epoch"] + 1
        self.best_score = checkpoint["best_score"]
        if self.optimizer: self.optimizer.load_state_dict(checkpoint["optimizer"])

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

    def _stft(self, tensor: torch.Tensor):
        batch, nchannel, nsample = tensor.size()
    
        tensor = tensor.reshape(batch*nchannel, nsample)

        tensor = torch.stft(input=tensor,
                        n_fft=self.config.model.n_fft,
                        hop_length=self.config.model.hop_length,
                        win_length=self.config.model.win_length,
                        window=torch.hann_window(window_length=self.config.model.win_length, dtype=tensor.dtype, device=tensor.device),
                        center=self.config.model.center,
                        pad_mode="reflect",
                        normalized=False, # *frame_length**(-0.5)
                        onesided=None,
                        return_complex=False,
                        )
        tensor /= self.config.model.win_length
        _, nfeature, nframe, ndtype = tensor.size()
        tensor = tensor.reshape(batch, nchannel, nfeature, nframe, ndtype)
        return tensor
        
    def _istft(self, tensor: torch.Tensor):
        batch, nchannel, nfeature, nframe, ndtype = tensor.size()
        tensor *= self.config.model.win_length
        tensor = tensor.reshape(batch*nchannel, nfeature, nframe, ndtype)
        tensor_complex = torch.complex(real=tensor[..., 0], imag=tensor[..., 1])
        
        tensor = torch.istft(
            input=tensor_complex,
            n_fft=self.config.model.n_fft,
            hop_length=self.config.model.hop_length,
            win_length=self.config.model.win_length,
            window=torch.hann_window(window_length=self.config.model.win_length, dtype=tensor.dtype, device=tensor.device),
            center=self.config.model.center,
            length=int(self.config.model.segment*self.config.model.sample_rate),
            normalized=False,
            onesided=None,
            return_complex=False,
        )
        _, nsample = tensor.size()
        tensor = tensor.reshape(batch, nchannel, nsample)
        return tensor

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

        for epoch in range(1, self.epochs+1):
            print(f"============== {epoch} / {self.epochs} epoch ==============")
            print("[0 seconds] Begin training...")
            
            start_time = time.time()

            score = self._run_one_epoch(epoch, self.epochs, train=True)
                        
            if epoch % self.save_checkpoint_interval == 0:
                self._save_checkpoint(epoch)

            if epoch % self.validation_interval == 0:
                print(f"[{int(time.time() - start_time)} seconds] Training is over, Validation is in progress...")
                score = self._run_one_epoch(epoch, self.epochs, train=False)
                if self._is_best(score, find_max=True):
                    self._save_checkpoint(epoch, is_best=True)
                    early_stopping = 0
                else:
                    early_stopping += 1
                
            if epoch % self.test_interval == 0:
                print(f"[{int(time.time() - start_time)} seconds] Training and Validation are over, Test is in progress...")
                score = self.inference(epoch, self.epochs)            

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
        name_dataset = self.config.default.dset.name
        loss_total = 0.
        grad_norm_total = 0.
        dataloader = self.train_dataloader if train else self.validation_dataloader

        total_step = len(dataloader)
        if (not self.config.solver.all_steps) and total_step > self.config.solver.total_steps:
            print(f"\tTotal step({self.config.solver.total_steps}) is less then the length of dataset({total_step})...")
            print(f"\tTrain will stop at step {self.config.solver.total_steps}!")
            total_step = self.config.solver.total_steps
    
        tepoch = tqdm.tqdm(dataloader, ncols=120) # [TODO] Search Pytorch progress bar 
        for step, batch in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            
            if step >= total_step:
                break

            if name_dataset == "Clarity":
                mixture, clean, interferer, mixture_metadata, clean_metadata, interferer_metadata, name, index = batch
            else:
                mixture, clean, mixture_metadata, clean_metadata, name, index = batch     

            mixture = mixture.to(self.device)
            clean = clean.to(self.device)
            if name_dataset == "Clarity": interferer = interferer.to(self.device)
            
            batch, nchannel, nsample = mixture.shape

            # mono channel to stereo for source separation models
            if self.config.model.name in ("demucs", "conv-tasnet") and nchannel == 1:
                # [TODO] shape
                try:
                    mixture = torch.concat(tensors=[mixture, mixture], dim=1)
                    clean = torch.concat(tensors=[clean, clean], dim=1)
                    if name_dataset == "Clarity": interferer = torch.concat(tensors=[clean, clean], dim=1)
                except AttributeError:
                    # For torch 1.7.1, AttributeError: module 'torch' has no attribute 'concat'
                    mixture = torch.cat(tensors=[mixture, mixture], dim=1)
                    clean = torch.cat(tensors=[clean, clean], dim=1)
                    if name_dataset == "Clarity": interferer = torch.cat(tensors=[interferer, interferer], dim=1)
            
            # if not source separation models, merge batch and channels
            if self.config.model.name not in ("demucs", "conv-tasnet", "unet"):
                mixture = torch.reshape(mixture, shape=(batch*nchannel, 1, nsample))
                clean = torch.reshape(clean, shape=(batch*nchannel, 1, nsample))
                if name_dataset == "Clarity": interferer = torch.reshape(interferer, shape=(batch*nchannel, 1, nsample))

            if self.config.model.name in ("mel-rnn", "dcunet", "crn", "dnn", "unet"):
                # Reference. https://espnet.github.io/espnet/_modules/espnet2/layers/stft.html
                # [batch, channel, nfeature, nframe, ndtype]
                mixture = self._stft(mixture)
                clean = self._stft(clean)
                if name_dataset == "Clarity": interferer = self._stft(interferer)

            if train:
                self.model.train()
            if not train:
                self.model.eval()
            
            # with torch.autograd.detect_anomaly(): # figuring out nan grads
            enhanced: torch.Tensor = self.model(mixture)

            # source separation models give out.shape = batch, sources, channels, features, 
            # currently sources is 1 and channels 2
            if self.config.model.name in ("demucs", "conv-tasnet") and enhanced.shape[1] == 1:
                enhanced = torch.squeeze(enhanced, dim=1)
            
            if self.config.optim.pit and name_dataset == "Clarity":
                with torch.no_grad():
                    index_enhanced, index_target = PermutationInvariantTraining(enhance=enhanced.detach().clone(), 
                                                                            target=[clean.detach().clone(), interferer.detach().clone()],
                                                                            loss_function=self.loss_function)

                loss: torch.Tensor = self.loss_function(enhanced[:, index_enhanced, ...], clean if index_target == 0 else interferer)
            else:
                loss: torch.Tensor = self.loss_function(clean, enhanced)

            if train:
                self.optimizer.zero_grad()
                loss.backward()

                if self.config.optim.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.optim.clip_grad)

                self.optimizer.step()

            loss = loss.detach().item()
            loss_total += loss

            grad_norm = 0
            for n, p in self.model.named_parameters():
                grad_norm += (p.grad.sum() ** 2)
            grad_norm = grad_norm.sqrt().item()
            grad_norm_total += grad_norm
            
            if not train:
                # with torch.no_grad():
                # # skip the metric for training speed
                # if self.config.model.name in ("mel-rnn", "dcunet", "crn", "dnn", "unet"):
                #     # Reference. https://espnet.github.io/espnet/_modules/espnet2/layers/stft.html
                #     mixture = self._istft(mixture)
                #     clean = self._istft(clean)    
                #     enhanced = self._istft(enhanced)    

                # if self.config.model.name not in ("demucs", "conv-tasnet"):
                #     mixture = torch.reshape(mixture, shape=(batch, nchannel, nsample))
                #     clean = torch.reshape(clean, shape=(batch, nchannel, nsample))
                #     enhanced = torch.reshape(enhanced, shape=(batch, nchannel, nsample))
                    
                # mixture = mixture.detach().cpu()
                # clean = clean.detach().cpu()
                # enhanced = enhanced.detach().cpu()

                # start, end = 0, 0
                # for ibatch in range(len(index)):
                #     end += index[ibatch]
                #     if self.config.dset.norm == "z-score":
                #         mix_std, mix_mean = mixture_metadata[ibatch]["std"], mixture_metadata[ibatch]["mean"]
                #         clean_std, clean_mean = clean_metadata[ibatch]["std"], clean_metadata[ibatch]["mean"]
                                                
                #         mixture[start:end] = mixture[start:end]*mix_std+mix_mean
                #         clean[start:end] = clean[start:end]*clean_std+clean_mean
                #         enhanced[start:end] = enhanced[start:end]*mix_std+mix_mean

                #     if self.config.dset.norm == "linear-scale":                        
                #         mix_min, mix_max = mixture_metadata[ibatch]["min"], mixture_metadata[ibatch]["max"]
                #         clean_min, clean_max = clean_metadata[ibatch]["min"], clean_metadata[ibatch]["max"]
                #         mixture[start:end] = mixture[start:end]*(mix_max-mix_min)+mix_min
                #         clean[start:end] = clean[start:end]*(clean_max-clean_min)+clean_min
                #         enhanced[start:end] = enhanced[start:end]*(mix_max-mix_min)+mix_min
                #     start += index[ibatch]
                                                    
                # score, score_reference = self.compute_metric(mixture=mixture, enhanced=enhanced, clean=clean)

                # for metric_name, metric_value in score.items():
                #     self.score[metric_name] += metric_value
                # for metric_name, metric_value in score_reference.items():
                #     self.score_reference[metric_name] += metric_value                

                # tepoch.set_postfix(loss=loss, metric=np.mean(self.score[self.config.solver.validation.metric]))
                self.writer.add_scalar(f"Validation/Loss_step", loss, (epoch+1)*total_step+step)
                self.writer.add_scalar(f"Validation/grad_norm_step", grad_norm, (epoch+1)*total_step+step)
                tepoch.set_postfix(loss_valid=loss)
            if train:
                self.writer.add_scalar(f"Train/Loss_step", loss, (epoch+1)*total_step+step)
                self.writer.add_scalar(f"Train/grad_norm_step", grad_norm, (epoch+1)*total_step+step)
                tepoch.set_postfix(loss_train=loss)

        if train:
            self.score["loss"] = loss_total / len(dataloader)        
            self.score["grad_norm"] = grad_norm_total / len(dataloader)        
            self.writer.add_scalar(f"Train/Loss", self.score["loss"], epoch)
            self.writer.add_scalar(f"Train/Grad_norm", self.score["grad_norm"], epoch)
        
        if not train:
            self.score["loss_valid"] = loss_total / len(dataloader)        
            self.score["grad_norm_valid"] = grad_norm_total / len(dataloader)        
            self.writer.add_scalar(f"Validation/Loss", self.score["loss_valid"], epoch)
            self.writer.add_scalar(f"Validation/Grad_norm", self.score["grad_norm_valid"], epoch)

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

    def inference(self, epoch, total_epoch):
        name_dataset = self.config.default.dset.name
        loss_total = 0.
        dataloader = self.test_dataloader

        total_step = len(dataloader)
        if (not self.config.solver.all_steps) and total_step > self.config.solver.test.total_steps:
            print(f"\tTotal step({self.config.solver.total_steps}) is less then the length of dataset({total_step})...")
            print(f"\tTrain will stop at step {self.config.solver.total_steps}!")
            total_step = self.config.solver.test.total_steps

        tepoch = tqdm.tqdm(dataloader, ncols=120)
        for step, batch in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            
            if step >= total_step:
                break
            
            if name_dataset == "Clarity":
                mixture, clean, interferer, origial_length, name = batch
            else:
                mixture, clean, origial_length, name = batch

            batch, nchannel, nsample = mixture.shape

            if self.config.model.name in ("demucs", "conv-tasnet") and nchannel == 1:
                try:
                    mixture = torch.cat(tensors=[mixture, mixture], dim=1)
                    clean = torch.cat(tensors=[clean, clean], dim=1)
                except AttributeError:
                    # For torch 1.7.1, AttributeError: module 'torch' has no attribute 'concat'
                    mixture = torch.cat(tensors=[mixture, mixture], dim=1)
                    clean = torch.cat(tensors=[clean, clean], dim=1)

            if self.config.model.name not in ("demucs", "conv-tasnet"):
                mixture = torch.reshape(mixture, shape=(batch*nchannel, 1, nsample))

            enhanced = evaluate(mixture=mixture, model=self.model, device=self.device, config=self.config)

            if self.config.model.name in ("demucs", "conv-tasnet"):
                enhanced = torch.squeeze(enhanced, dim=1)

            if self.config.model.name not in ("demucs", "conv-tasnet"):
                mixture = torch.reshape(mixture, shape=(batch, nchannel, nsample))
                enhanced = torch.reshape(enhanced, shape=(batch, nchannel, nsample))
            
            if self.config.optim.pit and name_dataset=="Clarity":
                with torch.no_grad():
                    index_enhance, _ = PermutationInvariantTraining(enhance=enhanced.detach().clone(),
                                                                target=[clean.detach().clone(), ],
                                                                loss_function=self.loss_function)
                enhanced = enhanced[:, index_enhance, ...]
            loss: torch.Tensor = self.loss_function(clean, enhanced)
            loss = loss.detach().item()
            loss_total += loss

            assert clean.shape == mixture.shape == enhanced.shape

            mixture = mixture.detach().cpu()
            clean = clean.detach().cpu()
            enhanced = enhanced.detach().cpu()
            score, score_reference = self.compute_metric(mixture=mixture, enhanced=enhanced, clean=clean)

            for metric_name, metric_value in score.items():
                self.score_inference[metric_name] += metric_value    
            for metric_name, metric_value in score_reference.items():
                self.score_inference_reference[metric_name] += metric_value  
            
            self.spec_audio_visualization(mixture=mixture, enhanced=enhanced, clean=clean, name=name[0], epoch=epoch)            

            if LIB_CLARITY and self.config.default.dset.name == 'Clarity':
                self.compute_metric_clarity(mixture=mixture, enhanced=enhanced, length=origial_length, name=name[0])
                tepoch.set_postfix(loss=loss, metric=np.mean(self.score_inference[self.config.solver.test.metric]), metric2=np.mean(self.score_inference["haspi"]))
            else:
                tepoch.set_postfix(loss=loss, metric=np.mean(self.score_inference[self.config.solver.test.metric]))

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

        batch, nchannel, nsamples = mixture.size()
        mixture_one_sequence = mixture.view(nchannel, batch, nsamples) 
        enhanced_one_sequence = enhanced.view(nchannel, batch, nsamples) 
        clean_one_sequence = clean.view(nchannel, batch, nsamples) 

        mixture_one_sequence = mixture_one_sequence.view(nchannel*batch*nsamples) 
        enhanced_one_sequence = enhanced_one_sequence.view(nchannel*batch*nsamples) 
        clean_one_sequence = clean_one_sequence.view(nchannel*batch*nsamples) 
    
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
    
        cfg = OmegaConf.load(self.config.default.dset.config)
        scene = name.split("_")[0]

        enhanced_resample = enhanced.clone().detach()
        mixture_resample = mixture.clone().detach()
        if self.config.dset.sample_rate != cfg.nalr.fs:
            enhanced_resample = julius.resample_frac(x=enhanced_resample, old_sr=self.config.dset.sample_rate, new_sr=cfg.nalr.fs,
                                                    output_length=None, full=True)
            mixture_resample = julius.resample_frac(x=mixture_resample, old_sr=self.config.dset.sample_rate, new_sr=cfg.nalr.fs,
                                                    output_length=None, full=True)
            
        enhanced_resample = torch.squeeze(enhanced_resample, dim=0).numpy()
        mixture_resample = torch.squeeze(mixture_resample, dim=0).numpy()

        score = evaluate_clarity(scene=scene, enhanced=enhanced_resample, sample_rate=cfg.nalr.fs, cfg=cfg)
        score_mixture = evaluate_clarity(scene=scene, enhanced=mixture_resample, sample_rate=cfg.nalr.fs, cfg=cfg)

        self.score_inference['haspi'].append(np.mean(score[0]))
        self.score_inference['hasqi'].append(np.mean(score[1]))
        self.score_inference_reference['haspi'].append(np.mean(score_mixture[0]))
        self.score_inference_reference['hasqi'].append(np.mean(score_mixture[1]))
        