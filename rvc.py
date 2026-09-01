#rvc.py
from multiprocessing import cpu_count, Pool, current_process
from pathlib import Path
import traceback

import torch
#from concurrent.futures import ThreadPoolExecutor
from scipy.io import wavfile
import numpy as np
import os
import sys

now_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(now_dir)
from infer.module.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from my_utils import load_audio
from infer.vc.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent

# Global variables for models to be loaded by workers
hubert_model_global = None
net_g_global = None
cpt_global = None
vc_global = None
version_global = None
config_global = None

class Config:
    def __init__(self, device, is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                print("16/10 series P40 forced single precision")
                self.is_half = False
                for config_file in ["32k.json", "40k.json", "48k.json"]:
                    with open(BASE_DIR / "src" / "configs" / config_file, "r") as f:
                        strr = f.read().replace("true", "false")
                    with open(BASE_DIR / "src" / "configs" / config_file, "w") as f:
                        f.write(strr)
                with open(BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open(BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open(BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open(BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
        elif torch.backends.mps.is_available():
            print("No N-card, use MPS")
            self.device = "mps"
        else:
            print("No N-card, use CPU")
            self.device = "cpu"
            self.is_half = True

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max


def process_chunk(args):
    # This function is executed by a worker process
    (
        audio_chunk,
        input_path,
        times,
        pitch_change,
        f0_method,
        index_path,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        rms_mix_rate,
        version,
        protect,
        crepe_hop_length,
        p_len
    ) = args
    
    return vc_global.pipeline(
        hubert_model_global,
        net_g_global,
        0,
        audio_chunk,
        input_path,
        times,
        pitch_change,
        f0_method,
        index_path,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        0,
        rms_mix_rate,
        version,
        protect,
        crepe_hop_length,
        p_len
    )

def worker_initializer(model_path, hubert_path, device, is_half):
    # This function is called once for each worker process
    global hubert_model_global, net_g_global, cpt_global, vc_global, version_global, config_global
    print(f"[{current_process().name}] Loading models...")
    
    try:
        config_global = Config(device, is_half)
        hubert_model_global = load_hubert(config_global.device, config_global.is_half, hubert_path)
        cpt_global, version_global, net_g_global, _, vc_global = get_vc(config_global.device, config_global.is_half, config_global, model_path)
        print(f"[{current_process().name}] Models loaded.")
    except Exception as e:
        print(f"[{current_process().name}] Error loading models: {e}")
        traceback.print_exc()
        raise

'''
def load_hubert(device, is_half, model_path):
    models, _, task = checkpoint_utils.load_model_ensemble_and_task([model_path], suffix='')
    hubert = models[0]
    hubert = hubert.to(device)

    if is_half:
        hubert = hubert.half()
    else:
        hubert = hubert.float()

    hubert.eval()
    return hubert
'''

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

    vc = Pipeline(tgt_sr, config)
    return cpt, version, net_g, tgt_sr, vc

def rvc_infer(index_path, index_rate, input_path, output_path, pitch_change, f0_method, cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model, rvc_model_path, hubert_model_path=os.path.join(os.getcwd(), 'infers', 'hubert_base.pt'), parrel_mode=False):
  if f0_method not in ['rmvpe', 'fcpe']:
    print("Warning: f0 method is not supported. Using 'rmvpe'.")
    f0_method = 'rmvpe'

  audio = load_audio(input_path, 16000)
  times = [0, 0, 0]

  if parrel_mode:
    print("PARREL MODE ACTIVATED.")

    if_f0 = cpt.get('f0', 1)

    if len(audio) / 16000 <= 60:
      print("Audio is shorter than 1 minute. Forcing single worker to avoid overhead.")
      num_workers = 1
    else:
      try:
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        prop = torch.cuda.get_device_properties(device)
        total_vram = prop.total_memory / 1024 / 1024 # MB
        
        model_size_mb = 0
        for param_name, param_tensor in cpt["weight"].items():
          model_size_mb += param_tensor.numel() * param_tensor.element_size() / 1024 / 1024
        model_size_mb += os.path.getsize(hubert_model_path) / 1024 / 1024
        
        vram_buffer_mb = 512
        num_workers = int((total_vram - vram_buffer_mb) / model_size_mb)
        num_workers = max(1, num_workers)
        num_workers = min(num_workers, cpu_count())
        print(f"Optimal number of workers: {num_workers} (Total VRAM: {total_vram:.2f}MB, Estimated Model size: {model_size_mb:.2f}MB)")
      except Exception as e:
        print(f"Could not determine VRAM. Falling back to CPU count. Error: {e}")
        num_workers = cpu_count()

    chunk_length = len(audio) // num_workers
    chunks = [audio[i * chunk_length:(i + 1) * chunk_length] for i in range(num_workers)]
    if len(audio) % num_workers != 0:
      chunks[-1] = np.concatenate((chunks[-1], audio[num_workers * chunk_length:]))

    args_list = [
      (hubert_model, net_g, 0, chunk, times, pitch_change, f0_method, index_path, index_rate, if_f0, tgt_sr, 0, rms_mix_rate, version, protect)
      for chunk in chunks
    ]

    def run_parallel_thread(idx):
      current_args = args_list[idx]
      return vc.pipeline(*current_args)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
      processed_chunks = list(executor.map(run_parallel_thread, range(num_workers)))
      
    audio_opt = np.concatenate(processed_chunks)
    
  else:
    if_f0 = cpt.get('f0', 1)
    p_len = audio.shape[0] // vc.window 
    audio_opt = vc.pipeline(hubert_model, net_g, 0, audio, times, pitch_change, f0_method, index_path, index_rate, if_f0, tgt_sr, 0, rms_mix_rate, version, protect)

  wavfile.write(output_path, tgt_sr, audio_opt)

