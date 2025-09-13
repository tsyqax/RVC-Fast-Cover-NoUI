# rvc.py
import torch.multiprocessing as mp
import torch
from pathlib import Path
from multiprocessing import cpu_count, Pool, shared_memory
import numpy as np
from fairseq import checkpoint_utils
from scipy.io import wavfile
import sys
import os
import librosa
import time

mp.set_start_method("spawn", force=True)

from infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from my_utils import load_audio
from vc_infer_pipeline import VC

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
now_dir = BASE_DIR / "src"
sys.path.append(str(now_dir))

class Config:
    def __init__(self, device_str, is_half):
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.device = self.device_config(device_str)
        self.x_pad, self.x_query, self.x_center, self.x_max = self._set_infer_params()

    def device_config(self, device_str):
        if "cuda" in device_str and torch.cuda.is_available():
            try:
                i_device = int(device_str.split(":")[-1])
                self.gpu_name = torch.cuda.get_device_name(i_device)
                if ("16" in self.gpu_name and "V100" not in self.gpu_name.upper()) or "P40" in self.gpu_name.upper() or "1060" in self.gpu_name or "1070" in self.gpu_name or "1080" in self.gpu_name:
                    print("16 series/10 series P40 forced single precision")
                    self.is_half = False
                
                try:
                    self.gpu_mem = int(torch.cuda.get_device_properties(i_device).total_memory / 1024 / 1024 / 1024 + 0.4)
                except Exception as e:
                    print(f"Warning: Could not get GPU memory properties: {e}")
                    self.gpu_mem = None
                return torch.device(device_str)
            except Exception as e:
                print(f"Warning: Could not parse CUDA device string '{device_str}': {e}")
                print("Falling back to CPU.")
                self.gpu_name = None
                self.gpu_mem = None
                return torch.device("cpu")
        elif device_str == "mps" and torch.backends.mps.is_available():
            print("No supported N-card found, use MPS for inference")
            self.gpu_name = None
            self.gpu_mem = None
            return torch.device("mps")
        else:
            print(f"Device '{device_str}' not recognized or not available. Using CPU for inference")
            self.is_half = True
            self.gpu_name = None
            self.gpu_mem = None
            return torch.device("cpu")

    def _set_infer_params(self) -> tuple:
        if self.n_cpu == 0:
            self.n_cpu = cpu_count()
        if self.is_half:
            x_pad, x_query, x_center, x_max = 3, 10, 60, 65
        else:
            x_pad, x_query, x_center, x_max = 1, 6, 38, 41
        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad, x_query, x_center, x_max = 1, 5, 30, 32
        return x_pad, x_query, x_center, x_max

hubert_model_worker, rvc_model_worker, vc_worker, net_g_worker, version_worker, tgt_sr_worker = None, None, None, None, None, None

def initialize_worker(model_paths_dict, config_dict_reverted):
    global hubert_model_worker, vc_worker, net_g_worker, version_worker, tgt_sr_worker
    
    device_str = config_dict_reverted['device']
    is_half = config_dict_reverted['is_half']
    hubert_model_path = Path(model_paths_dict['hubert'])
    rvc_model_path = Path(model_paths_dict['rvc'])

    try:
        device_obj = torch.device(device_str)
        hubert_model_worker = load_hubert(device_obj, is_half, hubert_model_path)
        cpt, version_worker, net_g_worker, tgt_sr_worker, vc_worker = get_vc(device_obj, is_half, Config(device_str, is_half), rvc_model_path)
        print(f"Models loaded successfully in worker {os.getpid()}")
    except Exception as e:
        print(f"Error loading models in worker initializer {os.getpid()}: {e}")
        raise RuntimeError(f"Failed to initialize worker models: {e}")

def process_chunk_task(chunk_info_with_gpu):
    global hubert_model_worker, vc_worker, net_g_worker, version_worker, tgt_sr_worker
    
    (shm_name, shm_shape, shm_dtype, input_path, times, pitch_change, f0_method, index_path,
     index_rate, if_f0, filter_radius, rms_mix_rate, protect, crepe_hop_length, index, crossfade_length, gpu_id) = chunk_info_with_gpu
    
    if hubert_model_worker is None or net_g_worker is None or vc_worker is None:
        raise RuntimeError(f"Models not loaded in worker {os.getpid()}. Initializer likely failed.")
        
    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    try:
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        audio_chunk = np.ndarray(shm_shape, dtype=shm_dtype, buffer=existing_shm.buf)

        result = vc_worker.pipeline(
            hubert_model_worker, net_g_worker, 0, audio_chunk, input_path, times, pitch_change,
            f0_method, index_path, index_rate, if_f0, filter_radius, tgt_sr_worker,
            0, rms_mix_rate, version_worker, protect, crepe_hop_length
        )

        existing_shm.close()
        return (index, result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Worker task error in process {os.getpid()} for chunk {index}: {e}")
        return (index, Exception(f"Worker task failed for chunk {index}"))

def load_hubert(device, is_half, model_path):
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([str(model_path)], suffix='')
    hubert = models[0]
    hubert = hubert.to(device)

    if is_half:
        hubert = hubert.half()
    else:
        hubert = hubert.float()
    hubert.eval()
    return hubert

def get_vc(device, is_half, config, model_path):
    cpt = torch.load(model_path, map_location='cpu')
    if "config" not in cpt or "weight" not in cpt:
        raise ValueError(f'Incorrect format for {model_path}. Use a voice model trained using RVC v2 instead.')

    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")

    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])

    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(device)

    if is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()

    vc = VC(tgt_sr, config)
    return cpt, version, net_g, tgt_sr, vc

def crossfade(audio1, audio2, duration):
    duration = min(duration, len(audio1), len(audio2))
    if duration == 0:
        return np.concatenate((audio1, audio2))

    fade_out = np.linspace(1, 0, duration)
    fade_in = np.linspace(0, 1, duration)

    if audio1.ndim == 2:
        fade_out = fade_out[:, np.newaxis]
        fade_in = fade_in[:, np.newaxis]

    audio1_fade_out = audio1[-duration:] * fade_out
    audio1_non_fade = audio1[:-duration]
    audio2_fade_in = audio2[:duration] * fade_in
    audio2_non_fade = audio2[duration:]

    combined_fade = audio1_fade_out + audio2_fade_in
    return np.concatenate((audio1_non_fade, combined_fade, audio2_non_fade))

def rvc_infer(index_path, index_rate, input_path, output_path, pitch_change, f0_method, cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model, rvc_model_input):
    if f0_method not in ['rmvpe', 'fcpe']:
        print(f"Warning: f0_method '{f0_method}' is not supported. Using 'rmvpe' instead.")
        f0_method = 'rmvpe'

    audio = load_audio(input_path, 16000)
    times = [0, 0, 0]
    if_f0 = cpt.get('f0', 1)

    chunk_length_sec = 15
    overlap_sec = 1
    chunk_length = int(chunk_length_sec * 16000)
    overlap_length = int(overlap_sec * 16000)
    crossfade_length = int(overlap_sec * tgt_sr)

    if len(audio) / 16000 > 60:
        print(f"Audio is longer than 60 seconds ({len(audio)/16000:.2f}s). Starting parallel processing.")

        chunks = []
        start = 0
        while start < len(audio):
            end = start + chunk_length
            chunk = audio[start:min(end + overlap_length, len(audio))]
            chunks.append(chunk)
            start += chunk_length

        chunk_tasks = []
        shm_list = []
        
        model_paths_for_init = {
            'hubert': str(BASE_DIR / 'src' / 'infers' / 'hubert_base.pt'),
            'rvc': str(rvc_model_input)
        }
        config_dict_for_init = {
            'device': str(vc.device),
            'is_half': vc.is_half
        }

        for i, chunk in enumerate(chunks):
            shm = shared_memory.SharedMemory(create=True, size=chunk.nbytes)
            chunk_np_array = np.ndarray(chunk.shape, dtype=chunk.dtype, buffer=shm.buf)
            chunk_np_array[:] = chunk[:]
            gpu_id = i % torch.cuda.device_count() if torch.cuda.is_available() else None
            chunk_tasks.append((shm.name, chunk.shape, chunk.dtype, input_path, times, pitch_change, f0_method, index_path,
                                 index_rate, if_f0, filter_radius, rms_mix_rate, protect, crepe_hop_length, i, crossfade_length, gpu_id))
            shm_list.append(shm)

        num_chunks = len(chunks)
        num_processes = min(cpu_count(), num_chunks)
        if torch.cuda.is_available():
            num_processes = min(torch.cuda.device_count(), num_processes)

        with Pool(processes=num_processes, initializer=initialize_worker, initargs=(model_paths_for_init, config_dict_for_init)) as pool:
            processed_results = pool.map(process_chunk_task, chunk_tasks)

        processed_chunks_with_index = []
        for result_tuple in processed_results:
            if isinstance(result_tuple, tuple) and len(result_tuple) == 2:
                index, result = result_tuple
                if isinstance(result, Exception):
                    pool.terminate()
                    pool.join()
                    for shm in shm_list:
                        try:
                            shm.close()
                            shm.unlink()
                        except FileNotFoundError:
                            pass
                    raise result
                processed_chunks_with_index.append((index, result))
            else:
                print(f"Warning: Unexpected result format from worker: {result_tuple}")
        
        for shm in shm_list:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass

        processed_chunks_with_index.sort(key=lambda x: x[0])

        audio_opt = processed_chunks_with_index[0][1]
        for i in range(1, num_chunks):
            current_chunk_result = processed_chunks_with_index[i][1]
            audio_opt = crossfade(audio_opt, current_chunk_result, crossfade_length)
    else:
        print(f"Audio is shorter than 60 seconds ({len(audio)/16000:.2f}s). Processing sequentially.")
        audio = audio.copy()
        audio_opt = vc.pipeline(
            hubert_model, net_g, 0, audio, input_path, times, pitch_change, f0_method,
            index_path, index_rate, if_f0, filter_radius, tgt_sr, 0, rms_mix_rate, version,
            protect, crepe_hop_length
        )

    wavfile.write(output_path, tgt_sr, audio_opt)
