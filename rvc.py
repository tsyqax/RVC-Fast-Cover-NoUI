import torch.multiprocessing as mp
import torch
from pathlib import Path
from multiprocessing import cpu_count
import numpy as np
from fairseq import checkpoint_utils
from scipy.io import wavfile
import sys
import os

# 멀티프로세싱 시작 방식 설정
# 윈도우에서는 'spawn'을, 리눅스/macOS에서는 'fork'를 사용합니다.
if sys.platform == "win32":
    mp.set_start_method("spawn", force=True)
else:
    mp.set_start_method("fork", force=True)

from infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from my_utils import load_audio
from vc_infer_pipeline import VC

BASE_DIR = Path(__file__).resolve().parent.parent

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
                print("16 series/10 series P40 forced single precision")
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
            print("No supported N-card found, use MPS for inference")
            self.device = "mps"
        else:
            print("No supported N-card found, use CPU for inference")
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

def worker(q_in, q_out, model_paths, config_dict):
    """
    Worker process 워커 프로세스.
    모델 객체 대신 경로를 받아 직접 로드하여 독립적으로 작동합니다.
    """
    try:
        device = config_dict['device']
        is_half = config_dict['is_half']
        hubert_model_path = model_paths['hubert']
        rvc_model_path = model_paths['rvc']

        hubert_model = load_hubert(device, is_half, hubert_model_path)
        cpt, version, net_g, tgt_sr, vc = get_vc(device, is_half, Config(device, is_half), rvc_model_path)

        while True:
            chunk_data = q_in.get()
            if chunk_data is None:
                break
            
            (audio_chunk, input_path, times, pitch_change, f0_method, index_path, 
             index_rate, if_f0, filter_radius, rms_mix_rate, protect, crepe_hop_length) = chunk_data
            
            result = vc.pipeline(
                hubert_model, net_g, 0, audio_chunk, input_path, times, pitch_change,
                f0_method, index_path, index_rate, if_f0, filter_radius, tgt_sr,
                0, rms_mix_rate, version, protect, crepe_hop_length
            )
            q_out.put((result, index))
            
        print("Worker process finished.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Worker process failed with an error: {e}")
        q_out.put(e)

def load_hubert(device, is_half, model_path):
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path], suffix='', )
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

def rvc_infer(index_path, index_rate, input_path, output_path, pitch_change, f0_method, cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model, rvc_model_input):
    if f0_method not in ['rmvpe', 'fcpe']:
        print(f"Warning: f0_method '{f0_method}' is not supported. Using 'rmvpe' instead.")
        f0_method = 'rmvpe'
    
    audio = load_audio(input_path, 16000)
    times = [0, 0, 0]
    if_f0 = cpt.get('f0', 1)

    if len(audio) / 16000 > 60:
        print("Audio is longer than 1 minute. Starting parallel processing.")
        
        num_chunks = max(2, int(len(audio) / 16000 / 60))
        chunk_length = len(audio) // num_chunks
        chunks = [audio[i * chunk_length:(i + 1) * chunk_length] for i in range(num_chunks)]
        if len(audio) % num_chunks != 0:
            chunks[-1] = np.concatenate((chunks[-1], audio[num_chunks * chunk_length:]))

        q_in = mp.Queue()
        q_out = mp.Queue()
        
        # 워커 프로세스로 전달할 모델 경로와 설정
        model_paths = {
            'hubert': os.path.join(os.getcwd(), 'infers', 'hubert_base.pt'),  # 실제 허브ert 모델 경로로 변경하세요!
            'rvc': rvc_model_input        # 실제 RVC 모델 경로로 변경하세요!
        }
        config_dict = {
            'device': vc.device,
            'is_half': vc.is_half
        }
        
        processes = [
            mp.Process(target=worker, args=(q_in, q_out, model_paths, config_dict))
            for _ in range(num_chunks)
        ]
        for p in processes:
            p.start()
        
        for i, chunk in enumerate(chunks):
            q_in.put((chunk, input_path, times, pitch_change, f0_method, index_path, index_rate, 
                      if_f0, filter_radius, rms_mix_rate, protect, crepe_hop_length, i))

        for _ in range(num_chunks):
            q_in.put(None)
            
        processed_chunks = []
        processed_chunks_with_index = []
        for _ in range(num_chunks):
            result, index = q_out.get()
            if isinstance(result, Exception):
                raise result
            processed_chunks_with_index.append((index, result))
        processed_chunks_with_index.sort(key=lambda x: x[0])
        
        audio_opt = np.concatenate([result for index, result in processed_chunks_with_index])

    else:
        # 1분 미만 오디오는 기존 방식대로 처리 (모델 객체를 인자로 사용)
        audio_opt = vc.pipeline(
            hubert_model, net_g, 0, audio, input_path, times, pitch_change, f0_method,
            index_path, index_rate, if_f0, filter_radius, tgt_sr, 0, rms_mix_rate, version,
            protect, crepe_hop_length
        )

    wavfile.write(output_path, tgt_sr, audio_opt)
