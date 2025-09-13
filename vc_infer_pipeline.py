# vc_pipeline.py
from functools import lru_cache
from time import time as ttime
import torch.nn.functional as F

import torch.nn as nn
from typing import Any
import faiss
import librosa
import numpy as np
import os
import parselmouth
import pyworld
import sys
import torch
import torch.nn.functional as F
import torchcrepe
import traceback
from scipy import signal
from torch import Tensor
from fcpe import FCPE # Assuming FCPE class is correctly imported from fcpe.py
from rmvpe import RMVPE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
now_dir = os.path.join(BASE_DIR, 'src')
sys.path.append(now_dir)

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

input_audio_path2wav = {}


def cache_harvest_f0(input_audio_path, fs, f0max, f0min, frame_period):
    audio = input_audio_path2wav[input_audio_path]
    f0, t = pyworld.harvest(
        audio,
        fs=fs,
        f0_ceil=f0max,
        f0_floor=f0min,
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(audio, f0, t, fs)
    return f0


def change_rms(data1, sr1, data2, sr2, rate):
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (
        torch.pow(rms1, torch.tensor(1 - rate))
        * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()
    return data2


class VC(object):
    def __init__(self, tgt_sr, config):
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half,
        )
        self.sr = 16000
        self.window = 160
        self.t_pad = self.sr * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query
        self.t_center = self.sr * self.x_center
        self.t_max = self.sr * self.x_max
        self.device = config.device
        # Initialize models here or load them on demand efficiently
        self.model_rmvpe = None
        self.model_fcpe = None # This will be an instance of FCPEInfer


    def get_optimal_torch_device(self, index: int = 0) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(
                f"cuda:{index % torch.cuda.device_count()}"
            )
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def get_f0(
        self,
        input_audio_path,
        x,
        p_len,
        f0_up_key,
        f0_method,
        filter_radius,
        crepe_hop_length,
        inp_f0=None,
    ):
        global input_audio_path2wav
        time_step = self.window / self.sr * 1000
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        f0 = None

        if f0_method == "rmvpe":
            if self.model_rmvpe is None:
                # Load RMVPE model if not already loaded
                self.model_rmvpe = RMVPE(
                    os.path.join(BASE_DIR, 'DIR', 'infers', 'rmvpe.pt'), is_half=self.is_half, device=self.device
                )
            # RMVPE infer_from_audio returns f0 directly
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        elif f0_method == "fcpe":
            # Load FCPE model (FCPEInfer) if not already loaded
            if self.model_fcpe is None:
                 # Assuming FCPEInfer needs model path, device, and dtype
                 # You might need to adjust the path and parameters based on your setup
                 # The FCPEInfer class from fcpe.py expects model_path, device, and dtype
                 try:
                     # Assuming fcpe.pt is the model file for FCPE
                     fcpe_model_path = os.path.join(BASE_DIR, 'DIR', 'infers', 'fcpe.pt') # Adjust path if needed
                     self.model_fcpe = FCPE(fcpe_model_path, device=self.device, dtype=torch.float16 if self.is_half else torch.float32)
                 except Exception as e:
                     print(f"Error loading FCPE model: {e}")
                     self.model_fcpe = None # Ensure model is None on failure
                     # Fallback to RMVPE if FCPE model loading fails
                     if self.model_rmvpe is None:
                          self.model_rmvpe = RMVPE(
                              os.path.join(BASE_DIR, 'DIR', 'infers', 'rmvpe.pt'), is_half=self.is_half, device=self.device
                          )
                     f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
                     print("Falling back to RMVPE for F0 extraction due to FCPE model loading error.")
                     return None, None # Return None if fallback happens due to loading error


            # Implement FCPE inference logic using the loaded model
            # The FCPEInfer.__call__ method takes audio (numpy array or torch tensor) and sample rate
            # It returns the F0 values.
            try:
                # Ensure audio data 'x' is in the correct format (numpy or tensor) and device for FCPEInfer
                # According to fcpe.py, FCPEInfer.__call__ takes audio as torch.FloatTensor
                x_tensor = torch.from_numpy(x).float().to(self.device) # Ensure data is float tensor on correct device
                # The FCPEInfer.__call__ method returns f0. Let's check its signature and output.
                # Based on fcpe.py, FCPEInfer.__call__ returns f0 as torch.Tensor [1, N, 1] or [1, N]
                # We need a numpy array of shape [N] for pitch and pitchf.
                f0_result = self.model_fcpe(x_tensor, sr=self.sr)
                if isinstance(f0_result, torch.Tensor):
                     f0 = f0_result.squeeze().cpu().numpy() # Convert tensor output to numpy array [N]
                else:
                     f0 = f0_result # Assume it's already numpy if not a tensor (less likely based on fcpe.py)

            except Exception as e:
                 print(f"Error during FCPE inference: {e}")
                 # Fallback to RMVPE on FCPE inference errors
                 if self.model_rmvpe is None:
                      self.model_rmvpe = RMVPE(
                          os.path.join(BASE_DIR, 'DIR', 'infers', 'rmvpe.pt'), is_half=self.is_half, device=self.device
                      )
                 f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
                 print("Falling back to RMVPE for F0 extraction due to FCPE inference error.")


        else:
            print(f"Warning: f0_method '{f0_method}' is not supported. Using 'rmvpe' instead.")
            if self.model_rmvpe is None:
                self.model_rmvpe = RMVPE(
                    os.path.join(BASE_DIR, 'DIR', 'infers', 'rmvpe.pt'), is_half=self.is_half, device=self.device
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)

        if f0 is None:
             # Handle cases where f0 could not be extracted by the chosen method or fallback
             print(f"Warning: F0 extraction failed with method '{f0_method}'. F0 will be treated as None.")
             return None, None # Return None for both pitch and pitchf

        # Ensure f0 is a numpy array before processing
        if not isinstance(f0, np.ndarray):
             f0 = np.array(f0)

        if isinstance(f0, float): # Handle case where f0 might still be a float after conversion
            f0 = np.array([f0])

        # Ensure f0 has at least one dimension if it's an empty array after conversion
        if f0.ndim == 0 and f0.size == 0:
             f0 = np.array([])


        f0bak = f0.copy()
        # Ensure f0 is not None before calculating f0_mel
        if f0 is not None and len(f0) > 0: # Added check for empty f0 array
            f0_mel = 1127 * np.log(1 + f0 / 700)
            f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
                f0_mel_max - f0_mel_min
            ) + 1
            f0_mel[f0_mel <= 1] = 1
            f0_mel[f0_mel > 255] = 255
            # Ensure the result is integer type
            f0_coarse = np.rint(f0_mel).astype(np.int)
        else:
            # Create an empty array with the expected shape if f0 is None or empty
            # Need to determine the expected length based on p_len or audio length
            # For now, create an empty array with int type
            f0_coarse = np.zeros_like(f0bak if f0bak is not None else [], dtype=np.int) # Handle case where f0bak might also be None

        return f0_coarse, f0bak

    def vc(
        self,
        model: nn.Module,
        net_g: nn.Module,
        sid: int,
        audio0: np.ndarray,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        times: list,
        index: Any,
        big_npy: np.ndarray,
        index_rate: float,
        version: str,
        protect: float,
    ):
        t0 = ttime()
        feats = torch.from_numpy(audio0)
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        inputs = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }

        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]

        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.clone()

        if (
            isinstance(index, type(None)) == False
            and isinstance(big_npy, type(None)) == False
            and index_rate != 0
        ):
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")
            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
            if self.is_half:
                npy = npy.astype("float16")
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )

        p_len = audio0.shape[0] // 320

        t1 = ttime()
        times[0] += t1 - t0

        with torch.no_grad():
            if pitch is not None and pitchf is not None:
                audio1 = (
                    (net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0])
                    .data.cpu()
                    .float()
                    .numpy()
                )
            else:
                audio1 = (
                    (net_g.infer(feats, p_len, sid)[0][0, 0]).data.cpu().float().numpy()
                )

        del feats, padding_mask

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        t2 = ttime()
        times[2] += t2 - t1
        return audio1

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        input_audio_path,
        times,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version,
        protect,
        crepe_hop_length,
        f0_file=None,
    ):
        # Note: This pipeline method currently handles the full audio.
        # For parallel processing, you would typically modify this method
        # or create a new one to process a single chunk of audio.

        if (
            file_index != ""
            and os.path.exists(file_index) == True
            and index_rate != 0
        ):
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except:
                traceback.print_exc()
                index = big_npy = None
        else:
            index = big_npy = None

        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None
        if hasattr(f0_file, "name") == True:
            try:
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0 = []
                for line in lines:
                    inp_f0.append([float(i) for i in line.split(",")])
                inp_f0 = np.array(inp_f0, dtype="float32")
            except:
                traceback.print_exc()

        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        if if_f0 == 1:
            pitch, pitchf = self.get_f0(
                input_audio_path,
                audio_pad,
                p_len,
                f0_up_key,
                f0_method,
                filter_radius,
                crepe_hop_length,
                inp_f0,
            )
            # Check if F0 extraction was successful
            if pitch is None or pitchf is None:
                 print("F0 extraction failed. Cannot proceed with inference.")
                 return None # Return None or raise an error

            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if self.device == "mps":
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()

        t2 = ttime()
        times[1] += t2 - t1

        # The original code had chunking logic here, which is removed to focus on single-pass inference within this method.
        # For parallel processing, the chunking/merging should be handled by the calling code.
        # This method will now assume it receives a full audio segment or a pre-determined chunk.

        # Perform VC on the (potentially padded) audio
        audio_opt = self.vc(
            model,
            net_g,
            sid,
            audio_pad, # Process the padded audio
            pitch,
            pitchf,
            times,
            index,
            big_npy,
            index_rate,
            version,
            protect,
        )[self.t_pad_tgt : -self.t_pad_tgt] # Remove padding from the output

        if rms_mix_rate != 1:
            # Note: If processing chunks, change_rms might need adjustment or applied after merging.
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)

        if resample_sr >= 16000 and tgt_sr != resample_sr:
            audio_opt = librosa.resample(
                audio_opt, orig_sr=tgt_sr, target_sr=resample_sr
            )
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_opt
