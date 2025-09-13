import numpy as np
import multiprocessing as mp
import gc
from vc_infer_pipeline import vc_infer_chunk

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

def crossfade(audio1, audio2, duration):
    # Simple crossfade for chunk merging
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

def infer_chunk(args):
    # chunk inference with index (for sorting)
    idx, chunk, model_args = args
    out_chunk = vc_infer_chunk(chunk, *model_args)
    return idx, out_chunk

def rvc_infer(index_path, index_rate, input_path, output_path, pitch_change, f0_method,
              cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect,
              crepe_hop_length, vc, hubert_model, rvc_model_input):
    # Load audio
    audio = load_audio(input_path, 16000)
    chunk_length_sec = 15
    overlap_sec = 1
    chunk_length = int(chunk_length_sec * 16000)
    overlap_length = int(overlap_sec * 16000)
    crossfade_length = int(overlap_sec * tgt_sr)

    if len(audio) / 16000 > 60:
        print("Parallel infer...")
        chunks = []
        starts = []
        start = 0
        while start < len(audio):
            end = start + chunk_length
            chunk = audio[start:min(end + overlap_length, len(audio))]
            chunks.append(chunk)
            starts.append(start)
            start += chunk_length
        model_args = [cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect,
                      crepe_hop_length, vc, hubert_model, rvc_model_input]
        tasks = [(idx, chunk, model_args) for idx, chunk in enumerate(chunks)]
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(infer_chunk, tasks)
        results.sort(key=lambda x: x[0])
        audio_chunks = [r[1] for r in results]
        out_audio = audio_chunks[0]
        for i in range(1, len(audio_chunks)):
            out_audio = crossfade(out_audio, audio_chunks[i], crossfade_length)
        save_audio(out_audio, output_path, tgt_sr)
    else:
        print("Serial infer...")
        out_audio = vc_infer_chunk(audio, cpt, version, net_g, filter_radius, tgt_sr,
                                  rms_mix_rate, protect, crepe_hop_length, vc,
                                  hubert_model, rvc_model_input)
        save_audio(out_audio, output_path, tgt_sr)
    gc.collect()
