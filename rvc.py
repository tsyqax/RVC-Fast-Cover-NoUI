from multiprocessing import cpu_count, Pool
from pathlib import Path
import torch
from fairseq import checkpoint_utils
from scipy.io import wavfile
from my_utils import load_audio, load_hubert, get_vc, Config

def process_chunk(args):
    (
        model_paths,
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
        vc,
        device,
        is_half,
        config
    ) = args
    
    hubert_model = load_hubert(device, is_half, model_paths["hubert"])
    cpt, _, net_g, _, vc_dummy = get_vc(device, is_half, config, model_paths["rvc"])
    
    return vc.pipeline(
        hubert_model, 
        net_g, 
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
        crepe_hop_length
    )

def rvc_infer(index_path, index_rate, input_path, output_path, pitch_change, f0_method, cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model):
    if f0_method not in ['rmvpe', 'fcpe']:
        f0_method = 'rmvpe'
    audio = load_audio(input_path, 16000)
    times = [0, 0, 0]
    if_f0 = cpt.get('f0', 1)
    if len(audio) / 16000 > 60:
        num_chunks = cpu_count()
        chunk_length = len(audio) // num_chunks
        chunks = [audio[i * chunk_length:(i + 1) * chunk_length] for i in range(num_chunks)]
        if len(audio) % num_chunks != 0:
            chunks[-1] = torch.cat((chunks[-1], audio[num_chunks * chunk_length:]))
        
        model_paths = {
            "hubert": "hubert_path",
            "rvc": "rvc_path",
        }
        
        args_list = [
            (
                model_paths,
                chunk,
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
                vc,
                hubert_model.device,
                hubert_model.is_half,
                vc.config
            )
            for chunk in chunks
        ]
        
        with Pool(cpu_count()) as p:
            processed_chunks = p.map(process_chunk, args_list)
        
        audio_opt = torch.cat(processed_chunks)
    else:
        audio_opt = vc.pipeline(hubert_model, net_g, 0, audio, input_path, times, pitch_change, f0_method, index_path, index_rate, if_f0, filter_radius, tgt_sr, 0, rms_mix_rate, version, protect, crepe_hop_length)

    wavfile.write(output_path, tgt_sr, audio_opt)
