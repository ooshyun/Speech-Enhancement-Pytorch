"""Compressor Class
This code converts torch tensor from https://github.com/claritychallenge/clarity
"""
from __future__ import annotations

from typing import Any, List, Tuple

import torch
from torch.nn.functional import pad
import numpy as np

class CompressorTorch:
    def __init__(
        self,
        fs: int = 44100,
        attack: int = 5,
        release: int = 20,
        threshold: int = 1,
        attenuation: float = 0.0001,
        rms_buffer_size: float = 0.2,
        makeup_gain: int = 1,
    ):
        """Instantiate the Compressor Class.

        Args:
            fs (int): (default = 44100)
            attack (int): (default = 5)
            release int: (default = 20)
            threshold (int): (default = 1)
            attenuation (float): (default = 0.0001)
            rms_buffer_size (float): (default = 0.2)
            makeup_gain (int): (default = 1)
        """
        self.fs = fs
        self.rms_buffer_size = rms_buffer_size
        self.set_attack(attack)
        self.set_release(release)
        self.threshold = threshold
        self.attenuation = attenuation
        self.eps = 1e-8
        self.makeup_gain = makeup_gain

        # window for computing rms
        self.win_len = int(self.rms_buffer_size * self.fs)
        self.window = np.ones(self.win_len)

    def set_attack(self, t_msec: float) -> None:
        """DESCRIPTION

        Args:
            t_msec (float): DESCRIPTION

        Returns:
            float: DESCRIPTION
        """
        t_sec = t_msec / 1000
        reciprocal_time = 1 / t_sec
        self.attack = reciprocal_time / self.fs

    def set_release(self, t_msec: float) -> None:
        """DESCRIPTION

        Args:
            t_msec (float): DESCRIPTION

        Returns:
            float: DESCRIPTION
        """
        t_sec = t_msec / 1000
        reciprocal_time = 1 / t_sec
        self.release = reciprocal_time / self.fs

    def process(self, signal: torch.Tensor) -> torch.Tensor:
        """DESCRIPTION

        Args:
            signal (np.array): DESCRIPTION

        Returns:
            np.array: DESCRIPTION
        """
        out = torch.zeros_like(signal)
        for batch in range(signal.shape[0]):
            for nspk in range(signal.shape[1]):
                signal_numpy = signal[batch, nspk].detach().cpu()
                padded_signal = np.concatenate((np.zeros(self.win_len - 1), signal_numpy), axis=-1)

                rms = np.sqrt(
                    np.convolve(padded_signal**2, self.window, mode="valid") / self.win_len
                    + self.eps
                )
                comp_ratios: list = []
                curr_comp: float = 1.0
                for rms_i in rms:
                    if rms_i > self.threshold:
                        temp_comp = (rms_i * self.attenuation) + (
                            (1 - self.attenuation) * self.threshold
                        )
                        curr_comp = (curr_comp * (1 - self.attack)) + (temp_comp * self.attack)
                    else:
                        curr_comp = (1 * self.release) + curr_comp * (1 - self.release)
                    comp_ratios.append(curr_comp)

                comp_ratios = torch.from_numpy(np.array(comp_ratios, dtype=np.float32))
                comp_ratios = comp_ratios.to(signal.device)

                out[batch, nspk] = signal[batch, nspk]*comp_ratios

        return out # assume makeup gain is 1
