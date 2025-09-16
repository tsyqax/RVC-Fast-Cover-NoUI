from functools import lru_cache
from time import time as ttime

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
from torchfcpe import spawn_bundled_infer_model
import traceback
from scipy import signal
from torch import Tensor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
now_dir = os.path.join(BASE_DIR, 'src')
sys.path.append(now_dir)

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

input_audio_path2wav = {}


@lru_cache
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
    # print(data1.max(),data2.max())
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

    # Fork Feature: Get the best torch device to use for f0 algorithms that require a torch device. Will return the type (torch.device)
    def get_optimal_torch_device(self, index: int = 0) -> torch.device:
        # Get cuda device
        if torch.cuda.is_available():
            return torch.device(
                f"cuda:{index % torch.cuda.device_count()}"
            )
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    # Fork Feature: Compute pYIN f0 method
    def get_f0_pyin_computation(self, x, f0_min, f0_max):
        y, sr = librosa.load("saudio/Sidney.wav", self.sr, mono=True)
        f0, _, _ = librosa.pyin(y, sr=self.sr, fmin=f0_min, fmax=f0_max)
        f0 = f0[1:]
        return f0

    # Fork Feature: Acquire median hybrid f0 estimation calculation
    def get_f0_hybrid_computation(
        self,
        methods_str,
        input_audio_path,
        x,
        f0_min,
        f0_max,
        p_len,
        filter_radius,
        crepe_hop_length,
        time_step,
    ):
        s = methods_str
        s = s.split("hybrid")[1]
        s = s.replace("[", "").replace("]", "")
        methods = s.split("+")
        f0_computation_stack = []

        print("f0 methods: %s" % str(methods))
        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)
        for method in methods:
            f0 = None
            if method == "pm":
                f0 = (
                    parselmouth.Sound(x, self.sr)
                    .to_pitch_ac(
                        time_step=time_step / 1000,
                        voicing_threshold=0.6,
                        pitch_floor=f0_min,
                        pitch_ceiling=f0_max,
                    )
                    .selected_array["frequency"]
                )
                pad_size = (p_len - len(f0) + 1) // 2
                if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                    f0 = np.pad(
                        f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                    )
            elif method == "crepe":
                f0 = self.get_f0_official_crepe_computation(x, f0_min, f0_max)
                f0 = f0[1:]
            elif method == "crepe-tiny":
                f0 = self.get_f0_official_crepe_computation(x, f0_min, f0_max, "tiny")
                f0 = f0[1:]
            elif method == "mangio-crepe":
                f0 = self.get_f0_crepe_computation(
                    x, f0_min, f0_max, p_len, crepe_hop_length
                )
            elif method == "mangio-crepe-tiny":
                f0 = self.get_f0_crepe_computation(
                    x, f0_min, f0_max, p_len, crepe_hop_length, "tiny"
                )
            elif method == "harvest":
                input_audio_path2wav[input_audio_path] = x.astype(np.double)
                f0 = cache_harvest_f0(input_audio_path, self.sr, f0_max, f0_min, 10)
                if filter_radius > 2:
                    f0 = signal.medfilt(f0, 3)
                f0 = f0[1:]
            elif method == "dio":
                f0, t = pyworld.dio(
                    x.astype(np.double),
                    fs=self.sr,
                    f0_ceil=f0_max,
                    f0_floor=f0_min,
                    frame_period=10,
                )
                f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.sr)
                f0 = signal.medfilt(f0, 3)
                f0 = f0[1:]
            f0_computation_stack.append(f0)

        for fc in f0_computation_stack:
            print(len(fc))

        print("hybrid median f0")
        f0_median_hybrid = None
        if len(f0_computation_stack) == 1:
            f0_median_hybrid = f0_computation_stack[0]
        else:
            f0_median_hybrid = np.nanmedian(f0_computation_stack, axis=0)
        return f0_median_hybrid

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

        if f0_method not in ['rmvpe', 'fcpe']:
             print(f"Warning: f0_method '{f0_method}' is not supported. Using 'rmvpe' instead.")
             f0_method = 'rmvpe'

        if f0_method == "rmvpe":
            if hasattr(self, "model_rmvpe") == False:
                from rmvpe import RMVPE

                self.model_rmvpe = RMVPE(
                    os.path.join(BASE_DIR, 'DIR', 'infers', 'rmvpe.pt'), is_half=self.is_half, device=self.device
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        elif f0_method == "fcpe":
            if not hasattr(self, "model_fcpe"):
                self.model_fcpe = spawn_bundled_infer_model(device=self.device)
                #self.model_fcpe.eval()

            hop_size = 160
            audio = librosa.to_mono(x.astype(np.float32))
            
            chunk_size = 16000 * 5
            all_f0 = []

            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                
                if len(chunk) < hop_size * 2:
                    continue
                
                audio_tensor = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(-1).to(self.device)
                
                with torch.no_grad():
                    f0_chunk = self.model_fcpe.infer(
                        audio_tensor, 
                        sr=16000, 
                        decoder_mode='local_argmax', 
                        threshold=0.006, 
                        f0_min=80, 
                        f0_max=880, 
                        interp_uv=False, 
                        output_interp_target_length=None
                    )
                
                all_f0.append(f0_chunk.detach().cpu().numpy().squeeze())
                
                del audio_tensor, f0_chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            f0 = np.concatenate(all_f0, axis=0)
            
            f0_target_length = (len(audio) // hop_size) + 1
            if len(f0) != f0_target_length:
                f0 = np.interp(
                    np.linspace(0, len(f0) - 1, f0_target_length),
                    np.arange(len(f0)),
                    f0
                )

        f0 *= pow(2, f0_up_key / 12)

        tf0 = self.sr // self.window
        if inp_f0 is not None:
            delta_t = np.round(
                (inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1
            ).astype("int16")
            replace_f0 = np.interp(
                list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1]
            )
            shape = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[
                :shape
            ]

        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int)

        return f0_coarse, f0bak

    def vc(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        times,
        index,
        big_npy,
        index_rate,
        version,
        protect,
        p_len
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
        t0 = ttime()
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
        if protect < 0.5 and pitch != None and pitchf != None:
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
        if protect < 0.5 and pitch != None and pitchf != None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
        t1 = ttime()
        times[0] += t1 - t0
        t2 = ttime()
        times[2] += t2 - t1
        with torch.no_grad():
            # p_len을 텐서로 변환
            p_len_tensor = torch.LongTensor([p_len]).to(self.device)
            if pitch != None and pitchf != None:
                audio1 = (
                    (net_g.infer(feats, p_len_tensor, pitch, pitchf, sid)[0][0, 0])
                    .data.cpu()
                    .float()
                    .numpy()
                )
            else:
                audio1 = (
                    (net_g.infer(feats, p_len_tensor, sid)[0][0, 0]).data.cpu().float().numpy()
                )
        del feats, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t2 = ttime()
        times[0] += t1 - t0
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
        p_len,
        f0_file=None,
    ):
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
        
        # 패딩 제거
        # audio = signal.filtfilt(bh, ah, audio) # bh, ah가 정의되지 않아 주석처리
        # audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        audio_pad = audio # 패딩을 제거하기 위해 원본 audio 사용
        
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.convolve(np.abs(audio), np.ones(self.window), 'valid')
            for t in range(self.t_center, audio.shape[0], self.t_center):
                audio_sum_idx = max(0, t - self.window // 2)
                local_sum = audio_sum[audio_sum_idx - self.t_query // 2 : audio_sum_idx + self.t_query // 2]
                if local_sum.size == 0:
                    continue
                min_index_local = np.argmin(local_sum)
                split_point = (audio_sum_idx - self.t_query // 2) + min_index_local
                opt_ts.append(split_point)
    
        s = 0
        audio_opt = []
        t = None
        t1 = ttime()
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
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if self.device == "mps":
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        
        t2 = ttime()
        times[1] += t2 - t1
        
        for t in opt_ts:
            t = t // self.window * self.window
            
            # B 코드의 개선된 오디오 분할 방식 적용
            audio_chunk = np.ascontiguousarray(audio[s:t])
            
            if if_f0 == 1:
                # B 코드의 개선된 피치 청크 슬라이싱 적용
                pitch_chunk = pitch[:, s // self.window : t // self.window]
                pitchf_chunk = pitchf[:, s // self.window : t // self.window]
                
                processed_audio = self.vc(
                    model,
                    net_g,
                    sid,
                    audio_chunk,
                    pitch_chunk,
                    pitchf_chunk,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                    len(audio_chunk) // self.window # B 코드의 문제 해결
                )
            else:
                processed_audio = self.vc(
                    model,
                    net_g,
                    sid,
                    audio_chunk,
                    None,
                    None,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                    len(audio_chunk) // self.window
                )
            
            audio_opt.append(processed_audio)
            s = t
            
        # 마지막 조각 처리
        audio_chunk_last = np.ascontiguousarray(audio[s:])
        if if_f0 == 1:
            pitch_chunk_last = pitch[:, s // self.window:]
            pitchf_chunk_last = pitchf[:, s // self.window:]
            processed_audio_last = self.vc(
                model,
                net_g,
                sid,
                audio_chunk_last,
                pitch_chunk_last,
                pitchf_chunk_last,
                times,
                index,
                big_npy,
                index_rate,
                version,
                protect,
                len(audio_chunk_last) // self.window
            )
        else:
            processed_audio_last = self.vc(
                model,
                net_g,
                sid,
                audio_chunk_last,
                None,
                None,
                times,
                index,
                big_npy,
                index_rate,
                version,
                protect,
                len(audio_chunk_last) // self.window
            )
            
        audio_opt.append(processed_audio_last)
        
        audio_opt = np.concatenate(audio_opt)
        
        if rms_mix_rate != 1:
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
