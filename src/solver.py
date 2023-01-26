"""
Most of parts are from https://github.com/haoxiangsnr/Wave-U-Net-for-Speech-Enhancement.git

Model list
----------
    'mel-rnn': MelRNN,
    'dccrn': DCCRN,
    'dcunet': DCUnet,
    'demucs': Demucs,
    'wav-unet': WavUnet,
    'conv-tasnet': ConvTasNet,
    'crn': CRN,

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
    - crn, nfeature 161, nfft 322

# wav-unet, dcunet, mel-rnn, 
# [TODO] Test crn, demucs, conv-tasnet, ddcrn 
"""
import os
import gc
import time
import json
import tqdm
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from shutil import copyfile

import datetime
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .metric import (
    WB_PESQ,
    STOI,
    SI_SDR,
)

from .utils import (
    prepare_device,
    obj2dict,
)

# from torchmetrics import __version__ as torchmetrics_version

# try:
#     from torchmetrics import ScaleInvariantSignalDistortionRatio
#     from torchmetrics.audio import (
#     ShortTimeObjectiveIntelligibility,
#     PerceptualEvaluationSpeechQuality,
# )
# except ImportError:
#     from torchmetrics import SI_SDR as ScaleInvariantSignalDistortionRatio

import sys

class Solver(object):
    def __init__(self, 
                config, 
                model, 
                optimizer, 
                loss_function,
                train_dataloader,
                validation_dataloader
                ):
        
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.n_gpu = torch.cuda.device_count()
        self.device = prepare_device(self.n_gpu, cudnn_deterministic=config.solver.cudnn_deterministic)

        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        os.environ['GPU_DEBUG']='0'

        self.optimizer = optimizer
        self.loss_function = loss_function

        self.model = model.to(self.device)
        
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(self.n_gpu)))

        # Trainer
        self.epochs = config.solver.epochs
        self.save_checkpoint_interval = config.solver.save_checkpoint_interval
        self.validation_interval = config.solver.validation.interval
        self.num_visualization = 0
        
        # The following args is not in the config file, We will update it if resume is True in later.
        self.score = {"best_score": -np.inf,
                    "loss": 0,
                    "stoi": 0,
                    "pesq": 0,
                    "sisdr": 0,}

        self.score_reference = {"loss": 0,
                    "stoi": 0,
                    "pesq": 0,
                    "sisdr": 0,}

        # self.metric_torch_reference = {"stoi": ShortTimeObjectiveIntelligibility(fs=config.model.sample_rate, extended=True),
        #             "pesq": PerceptualEvaluationSpeechQuality(fs=config.model.sample_rate, mode = "wb"),
        #             "sisdr": ScaleInvariantSignalDistortionRatio(zero_mean=False), }
        
        # self.metric_torch_estimation = {"stoi": ShortTimeObjectiveIntelligibility(fs=config.model.sample_rate, extended=True),
        #             "pesq": PerceptualEvaluationSpeechQuality(fs=config.model.sample_rate, mode = "wb"),
        #             "sisdr": ScaleInvariantSignalDistortionRatio(zero_mean=False), }
        

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
        
        print("Configurations are as follows: ")
        print(  obj2dict(config))        
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
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])

        print(f"Model checkpoint loaded.")

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

        print(f"Model preloaded successfully from {model_path.as_posix()}.")

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

    def _stft(self, tensor):
        batch, nchannel, nsample = tensor.size()
    
        tensor = tensor.view(batch*nchannel, nsample)

        tensor = torch.stft(input=tensor,
                        n_fft=self.config.model.n_fft,
                        hop_length=self.config.model.hop_length,
                        win_length=self.config.model.win_length,
                        window=torch.hann_window(window_length=self.config.model.win_length, dtype=tensor.dtype, device=tensor.device),
                        center=self.config.model.center,
                        pad_mode="reflect",
                        normalized=True, # *frame_length**(-0.5)
                        onesided=None,
                        return_complex=False,
                        )
        _, nfeature, nframe, ndtype = tensor.size()
        tensor = tensor.reshape(batch, nchannel, nfeature, nframe, ndtype)
        return tensor

        

    def _istft(self, tensor):
        batch, nchannel, nfeature, nframe, ndtype = tensor.size()

        tensor = tensor.view(batch*nchannel, nfeature, nframe, ndtype)

        tensor_complex = torch.complex(real=tensor[..., 0], imag=tensor[..., 1])

        tensor = torch.istft(
            input=tensor_complex,
            n_fft=self.config.model.n_fft,
            hop_length=self.config.model.hop_length,
            win_length=self.config.model.win_length,
            window=torch.hann_window(window_length=self.config.model.win_length, dtype=tensor.dtype, device=tensor.device),
            center=self.config.model.center,
            length=int(self.config.model.segment*self.config.model.sample_rate),
            normalized=True,
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
        """
        torch.save(state_dict, (self.checkpoints_dir / "latest_model.tar").as_posix())
        torch.save(state_dict["model"], (self.checkpoints_dir / f"model_{str(epoch).zfill(4)}_{self.config.solver.validation.metric}_{self.score['best_score']:2.8f}.pth").as_posix())
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
        for epoch in range(1, self.epochs+1):
            print(f"============== {epoch} / {self.epochs} epoch ==============")
            print("[0 seconds] Begin training...")
            
            start_time = time.time()

            score = self._run_one_epoch(epoch, self.epochs, train=True)
            
            print(f"[GPU Usage]: ", torch.cuda.memory_allocated(device=self.device))
            torch.cuda.empty_cache()
            
            if epoch % self.save_checkpoint_interval == 0:
                self._save_checkpoint(epoch)

            if epoch % self.validation_interval == 0:
                print(f"[{int(time.time() - start_time)} seconds] Training is over, Validation is in progress...")
                score = self._run_one_epoch(epoch, self.epochs, train=False)

                if self._is_best(score, find_max=True):
                    self._save_checkpoint(epoch, is_best=True)

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
            - crn, (channel, nfeature, nframe, 2) -> [TODO] (1, nfeature(161), nframe)
        """
        loss_total = 0.
        dataloader = self.train_dataloader if train else self.validation_dataloader
        with tqdm.tqdm(dataloader, ncols=120) as tepoch:
            for step, batch in enumerate(tepoch):
                if len(batch) == 4:
                    mixture, clean, name, index = batch
                else:
                    mixture, clean, mixture_metadata, clean_metadata, name, index = batch

                mixture = mixture.to(self.device)
                clean = clean.to(self.device)

                if self.config.model.name in ("mel-rnn", "dcunet", "crn"):
                    # Reference. https://espnet.github.io/espnet/_modules/espnet2/layers/stft.html
                    mixture = self._stft(mixture)
                    clean = self._stft(clean)                 

                tepoch.set_description(f"Epoch {epoch}")

                if train:
                    self.model.train()
                    enhanced: torch.Tensor = self.model(mixture)
                if not train:
                    self.model.eval()
                    enhanced: torch.Tensor = self.model(mixture)
                assert clean.shape == mixture.shape == enhanced.shape

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
                
                if not train:
                    with torch.no_grad():
                        mixture = mixture.detach()
                        clean = clean.detach()
                        enhanced = enhanced.detach()

                        if self.config.model.name in ("mel-rnn", "dcunet", "crn"):
                            mixture = self._istft(mixture)
                            clean = self._istft(clean)
                            enhanced = self._istft(enhanced)

                        if self.config.dset.norm == "z-score":
                            start = 0
                            for i in range(len(index)):
                                mixture_meta_mean, mixture_meta_std = mixture_metadata[i]["mean"], mixture_metadata[i]["std"]
                                clean_meta_mean, clean_meta_std = clean_metadata[i]["mean"], clean_metadata[i]["std"]
                                
                                mixture[start:start+index[i]] = mixture[start:start+index[i]]*mixture_meta_std+mixture_meta_mean
                                clean[start:start+index[i]] = clean[start:start+index[i]]*clean_meta_std+clean_meta_mean
                                enhanced[start:start+index[i]] = enhanced[start:start+index[i]]*mixture_meta_std+mixture_meta_mean
                                start += index[i]
                        
                        if self.config.dset.norm == "linear-scale":
                            start = 0
                            for i in range(len(index)):
                                mixture_meta_min, mixture_meta_max = mixture_metadata[i]["min"], mixture_metadata[i]["max"]
                                clean_meta_min, clean_meta_max = clean_metadata[i]["min"], clean_metadata[i]["max"]
                                
                                mixture[start:start+index[i]] = mixture[start:start+index[i]]*(mixture_meta_max-mixture_meta_min)+mixture_meta_min
                                clean[start:start+index[i]] = clean[start:start+index[i]]*(clean_meta_max-clean_meta_min)+clean_meta_min
                                enhanced[start:start+index[i]] = enhanced[start:start+index[i]]*(mixture_meta_max-mixture_meta_min)+mixture_meta_min
                                start += index[i]

                        mixture = mixture.cpu()
                        clean = clean.cpu()
                        enhanced = enhanced.cpu()

                        self.compute_metric(mixture=mixture, enhanced=enhanced, clean=clean, epoch=epoch)
                        self.spec_audio_visualization(mixture=mixture, enhanced=enhanced, clean=clean, names=name, index=index, epoch=epoch)

        if train:
            tepoch.set_postfix(loss=loss)

            self.score["loss"] = loss_total / len(dataloader)        
            self.writer.add_scalar(f"Train/Loss", self.score["loss"], epoch)
        
        if not train:        
            tepoch.set_postfix(loss=loss, metric=self.score[self.config.solver.validation.metric] / (step+1))

            length_dataloader = len(dataloader)
            for metric in list(self.metric.keys()):
                self.score_reference[metric] = self.score_reference[metric] / length_dataloader
                self.score[metric] = self.score[metric] / length_dataloader
                
                self.writer.add_scalars(f"Validation/{metric}", {
                    "clean and noisy": self.score_reference[metric],
                    "clean and enhanced": self.score[metric],
                }, epoch)

                # self.writer.add_scalars(f"Validation/{metric}_torch", {
                #     "clean and noisy": self.metric_torch_reference[metric].compute(),
                #     "clean and enhanced": self.metric_torch_estimation[metric].compute(),
                # }, epoch)

        return self.score["loss"] if train else self.score[self.config.solver.validation.metric]

    def spec_audio_visualization(self, mixture, enhanced, clean, names, index, epoch):
        # Visualize audio
        if self.num_visualization > self.config.solver.validation.num_show:
            return

        start_index = 0
        end_index = 0
        for i, name in enumerate(names):
            if self.num_visualization > self.config.solver.validation.num_show:
                break
            end_index += index[i]
            
            mixture_one_sequence = mixture[start_index:end_index, ...]
            enhanced_one_sequence = enhanced[start_index:end_index, ...]
            clean_one_sequence = clean[start_index:end_index, ...]

            batch, nchannel, nsamples = mixture_one_sequence.size()
            mixture_one_sequence = mixture_one_sequence.view(nchannel, batch, nsamples) 
            enhanced_one_sequence = enhanced_one_sequence.view(nchannel, batch, nsamples) 
            clean_one_sequence = clean_one_sequence.view(nchannel, batch, nsamples) 

            mixture_one_sequence = mixture_one_sequence.view(nchannel*batch*nsamples) 
            enhanced_one_sequence = enhanced_one_sequence.view(nchannel*batch*nsamples) 
            clean_one_sequence = clean_one_sequence.view(nchannel*batch*nsamples) 
        
            self.writer.add_audio(f"Speech/{name}_mixture", mixture_one_sequence, epoch, sample_rate=16000)
            self.writer.add_audio(f"Speech/{name}_Enhanced", enhanced_one_sequence, epoch, sample_rate=16000)
            self.writer.add_audio(f"Speech/{name}_Clean", clean_one_sequence, epoch, sample_rate=16000)
            
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
                librosa.display.waveshow(y, sr=16000, ax=ax[j])
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

            start_index += index[i]
            self.num_visualization += 1

    def compute_metric(self, mixture, enhanced, clean, epoch):
        metric_name = list(self.metric.keys())
        for metric in metric_name:
            score_mixture = []
            score_enhanced = []

            # self.metric_torch_reference[metric].update(preds=mixture, target=clean)
            # self.metric_torch_estimation[metric].update(preds=enhanced, target=clean)
            score_mixture.append(self.metric[metric](estimation=mixture, reference=clean))
            score_enhanced.append(self.metric[metric](estimation=enhanced, reference=clean))

            self.score_reference[metric] += np.mean(score_mixture)
            self.score[metric] += np.mean(score_enhanced)
