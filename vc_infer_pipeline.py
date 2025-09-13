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
from scipy import signal
from torch import Tensor
from fcpe import FCPE
from rmvpe import RMVPE
from pathlib import Path
import traceback

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
now_dir = BASE_DIR / "DIR"
sys.path.append(str(now_dir))

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)


@lru_cache(maxsize=1024)
def get_rmvpe(is_half, device):
    try:
        model_path = BASE_DIR / "DIR" / "infers" / "rmvpe.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"RMVPE model not found at {model_path}")
        return RMVPE(str(model_path), is_half=is_half, device=device)
    except Exception as e:
        print(f"Failed to load RMVPE model: {e}")
        return None


@lru_cache(maxsize=1024)
def get_fcpe(is_half, device):
    try:
        model_path = BASE_DIR / "DIR" / "infers" / "fcpe.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"FCPE model not found at {model_path}")
        return FCPE(str(model_path), device=device, dtype=torch.float16 if is_half else torch.float32)
    except Exception as e:
        print(f"Failed to load FCPE model: {e}")
        return None


def change_rms(data1, sr1, data2, sr2, rate):
    rms1 = librosa.feature.rms(y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2)
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(rms1.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(rms2.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (torch.pow(rms1, torch.tensor(1 - rate)) * torch.pow(rms2, torch.tensor(rate - 1))).numpy()
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
        self.model_rmvpe = None
        self.model_fcpe = None

    def get_f0(
        self,
        x: np.ndarray,
        p_len: int,
        f0_up_key,
        f0_method: str,
        filter_radius,
        crepe_hop_length,
    ):
        f0_min = 50
        f0_max = 1100
        
        if f0_method == "rmvpe":
            if self.model_rmvpe is None:
                self.model_rmvpe = get_rmvpe(self.is_half, self.device)
                if self.model_rmvpe is None:
                    raise RuntimeError("RMVPE model is not loaded.")
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
            
        elif f0_method == "fcpe":
            if self.model_fcpe is None:
                self.model_fcpe = get_fcpe(self.is_half, self.device)
                if self.model_fcpe is None:
                    print("FCPE model not found. Falling back to RMVPE.")
                    self.model_rmvpe = get_rmvpe(self.is_half, self.device)
                    if self.model_rmvpe is None:
                        raise RuntimeError("Neither FCPE nor RMVPE model could be loaded.")
                    f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
                else:
                    x_tensor = torch.from_numpy(x).float().to(self.device)
                    f0 = self.model_fcpe(x_tensor, sr=self.sr).squeeze().cpu().numpy()
        else:
            print(f"Warning: f0_method '{f0_method}' is not supported. Using 'rmvpe' instead.")
            self.model_rmvpe = get_rmvpe(self.is_half, self.device)
            if self.model_rmvpe is None:
                raise RuntimeError("RMVPE model is not loaded.")
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)

        if f0 is None or len(f0) == 0:
            print("F0 extraction failed. Returning empty arrays.")
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        f0bak = f0.copy()
        
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int64)

        return f0_coarse, f0bak

    def vc(
        self,
        model: nn.Module,
        net_g: nn.Module,
        sid: int,
        audio0: np.ndarray,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        index: Any,
        big_npy: np.ndarray,
        index_rate: float,
        version: str,
        protect: float,
    ):
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

        if isinstance(index, type(None)) == False and isinstance(big_npy, type(None)) == False and index_rate != 0:
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")
            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
            if self.is_half:
                npy = npy.astype("float16")
            feats = torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate + (1 - index_rate) * feats

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        p_len = audio0.shape[0] // 320

        with torch.no_grad():
            if pitch is not None and pitchf is not None:
                audio1 = (net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0]).data.cpu().float().numpy()
            else:
                audio1 = (net_g.infer(feats, p_len, sid)[0][0, 0]).data.cpu().float().numpy()
        
        del feats, padding_mask
        if protect < 0.5 and pitch is not None and pitchf is not None:
            del feats0

        return audio1

    def vc_infer_chunk(chunk, cpt, version, net_g, filter_radius, tgt_sr,
                   rms_mix_rate, protect, crepe_hop_length, vc, hubert_model, rvc_model_input):
        out_audio = vc.infer(
            chunk, net_g, hubert_model,
            filter_radius, tgt_sr,
            rms_mix_rate, protect,
            crepe_hop_length, rvc_model_input
        )
        return out_audio
    
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
        t0 = ttime()

        index = big_npy = None
        if file_index and Path(file_index).exists() and index_rate != 0:
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception:
                traceback.print_exc()

        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None
        if f0_file and hasattr(f0_file, "name"):
            try:
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0 = np.array([[float(i) for i in line.split(",")] for line in lines], dtype="float32")
            except Exception:
                traceback.print_exc()

        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        
        if if_f0 == 1:
            try:
                pitch, pitchf = self.get_f0(
                    audio_pad, p_len, f0_up_key, f0_method, filter_radius, crepe_hop_length
                )
            except Exception as e:
                print(f"F0 extraction failed: {e}. Cannot proceed with inference.")
                return None
            
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len].astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
            
        t1 = ttime()
        
        audio_opt = self.vc(
            model,
            net_g,
            sid,
            audio_pad,
            pitch,
            pitchf,
            index,
            big_npy,
            index_rate,
            version,
            protect,
        )[self.t_pad_tgt : -self.t_pad_tgt]
        
        t2 = ttime()

        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)

        if resample_sr >= 16000 and tgt_sr != resample_sr:
            audio_opt = librosa.resample(audio_opt, orig_sr=tgt_sr, target_sr=resample_sr)
        
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return audio_opt
