#rvc.py
import torch.multiprocessing as mp
import torch
from pathlib import Path
from multiprocessing import cpu_count, Pool, shared_memory # Import shared_memory
import numpy as np
from fairseq import checkpoint_utils
from scipy.io import wavfile
import sys
import os
import librosa # Add librosa for overlap and crossfade
import time # Import time for potential sleep in workers if needed

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
                for config_file in [BASE_DIR / "src" / "configs" / "32k.json", BASE_DIR / "src" / "configs" / "40k.json", BASE_DIR / "src" / "configs" / "48k.json"]:
                    try:
                        with open(config_file, "r") as f:
                            strr = f.read().replace("true", "false")
                        with open(config_file, "w") as f:
                            f.write(strr)
                    except FileNotFoundError:
                        print(f"Warning: Config file not found at {config_file}")

                try:
                    with open(BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py", "r") as f:
                        strr = f.read().replace("3.7", "3.0")
                    with open(BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py", "w") as f:
                        f.write(strr)
                except FileNotFoundError:
                    print("Warning: trainset_preprocess_pipeline_print.py not found.")

            else:
                self.gpu_name = None
            try:
                self.gpu_mem = int(
                    torch.cuda.get_device_properties(i_device).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
                if self.gpu_mem is not None and self.gpu_mem <= 4: # Check if gpu_mem is not None before comparison
                    try:
                        with open(BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py", "r") as f:
                            strr = f.read().replace("3.7", "3.0")
                        with open(BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py", "w") as f:
                            f.write(strr)
                    except FileNotFoundError:
                         print("Warning: trainset_preprocess_pipeline_print.py not found.")
            except Exception as e:
                 print(f"Warning: Could not get GPU memory properties: {e}")
                 self.gpu_mem = None # Ensure gpu_mem is None if an error occurs


        elif torch.backends.mps.is_available():
            print("No supported N-card found, use MPS for inference")
            self.device = "mps"
        else:
            print("No supported N-card found, use CPU for inference")
            self.device = "cpu"
            self.is_half = True # CPU inference is typically done in full precision

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

        if self.gpu_mem is not None and self.gpu_mem <= 4: # Check if gpu_mem is not None before comparison
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max

# Global variables to hold models loaded in the initializer
global hubert_model_worker, rvc_model_worker, vc_worker, net_g_worker, version_worker, tgt_sr_worker, shm_worker

hubert_model_worker = None
rvc_model_worker = None
vc_worker = None
net_g_worker = None
version_worker = None
tgt_sr_worker = None
shm_worker = None # Shared memory handle in the worker

def worker_initializer(model_paths, config_dict, gpu_id=None): # Added gpu_id
    """
    Initializer function for the multiprocessing Pool.
    Loads models once per worker process.
    Optionally sets GPU device for the worker.
    """
    global hubert_model_worker, rvc_model_worker, vc_worker, net_g_worker, version_worker, tgt_sr_worker
    device_str = config_dict['device']
    is_half = config_dict['is_half']
    hubert_model_path = model_paths['hubert']
    rvc_model_path = model_paths['rvc']

    # Set GPU device for the worker if gpu_id is provided and CUDA is available
    if gpu_id is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
            config_dict['device'] = device # Update device in config_dict for VC init
            print(f"Worker process initialized on GPU {gpu_id}")
        except Exception as e:
            print(f"Warning: Could not set GPU {gpu_id} for worker: {e}")
            # Fallback to default device or handle as needed
            device = torch.device(device_str) # Use original device string
    else:
        device = torch.device(device_str) # Use original device string if no gpu_id or CUDA not available

    try:
        # Load models using the potentially updated device
        hubert_model_worker = load_hubert(device, is_half, hubert_model_path)
        cpt, version_worker, net_g_worker, tgt_sr_worker, vc_worker = get_vc(device, is_half, Config(device, is_half), rvc_model_path)
    except Exception as e:
        print(f"Error loading models in worker initializer: {e}")
        hubert_model_worker = None
        rvc_model_worker = None # Ensure these are None on failure

def worker_task(chunk_info):
    """
    Task function executed by each worker process.
    Receives shared memory info and processes the chunk.
    """
    global hubert_model_worker, rvc_model_worker, vc_worker, net_g_worker, version_worker, tgt_sr_worker, shm_worker

    if hubert_model_worker is None or net_g_worker is None:
         print("Models not loaded in worker. Cannot process chunk.")
         # Return index and an error indicator or exception
         if len(chunk_info) > 0 and isinstance(chunk_info[-2], int): # Check if index is the second to last element
             return (chunk_info[-2], Exception("Models not loaded in worker"))
         else:
             return (None, Exception("Models not loaded in worker")) # Cannot return index if not provided

    try:
        # Unpack chunk info including shared memory details and index
        (shm_name, shm_shape, shm_dtype, input_path, times, pitch_change, f0_method, index_path,
         index_rate, if_f0, filter_radius, rms_mix_rate, protect, crepe_hop_length, index, crossfade_length) = chunk_info

        # Attach to the shared memory and reconstruct the numpy array
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        # Reconstruct the numpy array from shared memory
        audio_chunk = np.ndarray(shm_shape, dtype=shm_dtype, buffer=existing_shm.buf)

        # Perform VC inference using the pre-loaded models
        # TODO: Modify VC class in vc_infer_pipeline.py to handle chunks efficiently
        result = vc_worker.pipeline(
            hubert_model_worker, net_g_worker, 0, audio_chunk, input_path, times, pitch_change,
            f0_method, index_path, index_rate, if_f0, filter_radius, tgt_sr_worker,
            0, rms_mix_rate, version_worker, protect, crepe_hop_length
        )

        # Explicitly close the shared memory handle in the worker
        existing_shm.close()

        # Return result with its original index
        return (index, result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"WORKER TASK ERROR: {e}")
        # Return exception and index for main process to handle
        if len(chunk_info) > 0 and isinstance(chunk_info[-2], int): # Check if index is the second to last element
             return (chunk_info[-2], e)
        else:
             return (None, e) # Cannot return index if not provided


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

def crossfade(audio1, audio2, duration):
    """Applies a linear crossfade between the end of audio1 and the beginning of audio2."""
    # Ensure duration is not longer than the audio segments
    duration = min(duration, len(audio1), len(audio2))

    if duration == 0:
        return np.concatenate((audio1, audio2))

    fade_in_start = len(audio2) - duration
    fade_out_end = duration

    # Create fade curves
    fade_out = np.linspace(1, 0, duration)
    fade_in = np.linspace(0, 1, duration)

    # Ensure fade arrays have the correct shape for stereo audio if applicable
    if audio1.ndim == 2:
        fade_out = fade_out[:, np.newaxis]
        fade_in = fade_in[:, np.newaxis]


    # Apply fade out to the end of audio1
    audio1_fade_out = audio1[-duration:] * fade_out
    audio1_non_fade = audio1[:-duration]

    # Apply fade in to the beginning of audio2
    audio2_fade_in = audio2[:duration] * fade_in
    audio2_non_fade = audio2[duration:]

    # Combine the faded sections
    combined_fade = audio1_fade_out + audio2_fade_in

    # Concatenate the non-faded parts and the combined fade
    return np.concatenate((audio1_non_fade, combined_fade, audio2_non_fade))


def rvc_infer(index_path, index_rate, input_path, output_path, pitch_change, f0_method, cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model, rvc_model_input):
    if f0_method not in ['rmvpe', 'fcpe']:
        print(f"Warning: f0_method '{f0_method}' is not supported. Using 'rmvpe' instead.")
        f0_method = 'rmvpe'

    audio = load_audio(input_path, 16000)
    times = [0, 0, 0]
    if_f0 = cpt.get('f0', 1)

    # Define chunk length and overlap
    # Adjust these values based on your audio and desired processing
    chunk_length_sec = 15 # Process audio in 15-second chunks (at 16000 Hz)
    overlap_sec = 1 # 1 second overlap
    chunk_length = int(chunk_length_sec * 16000) # Chunk length in samples (source SR)
    overlap_length = int(overlap_sec * 16000) # Overlap length in samples (source SR)
    crossfade_length = int(overlap_sec * tgt_sr) # Crossfade length in samples (target SR) - Use overlap_sec for consistency

    if len(audio) / 16000 > chunk_length_sec: # Use chunk_length_sec for comparison
        print(f"Audio is longer than {chunk_length_sec} seconds. Starting parallel processing.")

        # Split audio into chunks with overlap
        chunks = []
        start = 0
        while start < len(audio):
            end = start + chunk_length
            # Add overlap to the end of the chunk, but don't go beyond audio length
            chunk = audio[start:min(end + overlap_length, len(audio))]
            chunks.append(chunk)
            start += chunk_length # Move start by chunk length

        # Prepare chunk data including index and crossfade length for worker tasks
        chunk_tasks = []
        shm_list = [] # Keep track of shared memory blocks to unlink them
        for i, chunk in enumerate(chunks):
             # Create a shared memory block for each chunk
             shm = shared_memory.SharedMemory(create=True, size=chunk.nbytes)
             # Copy the chunk data to shared memory
             chunk_np_array = np.ndarray(chunk.shape, dtype=chunk.dtype, buffer=shm.buf)
             chunk_np_array[:] = chunk[:] # Copy data

             # Store shared memory info, index, and other parameters
             # Pass gpu_id to the worker_initializer if CUDA is available
             gpu_id = i % torch.cuda.device_count() if torch.cuda.is_available() else None
             chunk_tasks.append((shm.name, chunk.shape, chunk.dtype, input_path, times, pitch_change, f0_method, index_path,
                                  index_rate, if_f0, filter_radius, rms_mix_rate, protect, crepe_hop_length, i, crossfade_length, gpu_id))
             shm_list.append(shm)


        # Prepare model paths and config for the initializer
        model_paths = {
            'hubert': os.path.join(os.getcwd(), 'infers', 'hubert_base.pt'),  # 실제 허버트 모델 경로로 변경하세요!
            'rvc': rvc_model_input        # 실제 RVC 모델 경로로 변경하세요!
        }
        config_dict = {
            'device': vc.device, # Use device from the config object created in main
            'is_half': vc.is_half # Use is_half from the config object created in main
        }


        num_chunks = len(chunks)
        num_processes = min(cpu_count(), num_chunks) # Use min of CPU count and num chunks
        # If using GPU, limit num_processes to the number of GPUs
        if torch.cuda.is_available():
            num_processes = min(torch.cuda.device_count(), num_processes)


        # Use multiprocessing Pool with initializer
        # The initializer loads the models once per process and sets GPU
        # Pass shared memory info and index, and gpu_id to worker_task via pool.map
        # worker_initializer requires model_paths, config_dict, and gpu_id
        # We need to pass the gpu_id to the initializer for each worker.
        # This is tricky with pool.map. A common approach is to create separate initargs for each worker
        # or pass the gpu_id along with the chunk info and handle it in worker_task (less ideal for init).
        # Let's adjust worker_initializer to take gpu_id and pass it to Config.

        # A simpler approach with pool.map is to pass the gpu_id in the chunk_info
        # and have the worker_task handle setting the device, or use a dedicated initializer for each worker.
        # Let's refine the worker_initializer to accept gpu_id. The Pool will create processes and run the initializer.

        # We need a way to assign a unique gpu_id to each process created by the pool.
        # The initializer runs *after* the process is forked/spawned.
        # We can't easily pass a *specific* gpu_id to a *specific* process via initargs in pool.map in this way.

        # Let's revert the worker_initializer signature and pass the gpu_id in chunk_info
        # and set the device at the beginning of worker_task. This is a common workaround
        # when using pool.map with per-process configuration like GPU assignment.

        # Revert worker_initializer signature
        def worker_initializer_reverted(model_paths_dict, config_dict_reverted):
            """
            Initializer function for the multiprocessing Pool.
            Loads models once per worker process.
            GPU setting will be handled in worker_task.
            """
            global hubert_model_worker, rvc_model_worker, vc_worker, net_g_worker, version_worker, tgt_sr_worker
            device_str = config_dict_reverted['device']
            is_half = config_dict_reverted['is_half']
            hubert_model_path = model_paths_dict['hubert']
            rvc_model_path = model_paths_dict['rvc']

            try:
                # Load models using the original device string
                hubert_model_worker = load_hubert(torch.device(device_str), is_half, hubert_model_path)
                # Need to create Config with the *correct* device for VC. Let's pass device_str to Config too.
                cpt, version_worker, net_g_worker, tgt_sr_worker, vc_worker = get_vc(torch.device(device_str), is_half, Config(torch.device(device_str), is_half), rvc_model_path)
            except Exception as e:
                print(f"Error loading models in worker initializer: {e}")
                hubert_model_worker = None
                rvc_model_worker = None


        # Update worker_task to set GPU device
        def worker_task_with_gpu(chunk_info_with_gpu):
            """
            Task function executed by each worker process.
            Receives shared memory info, chunk index, crossfade length, and gpu_id.
            Sets GPU device at the start.
            """
            global hubert_model_worker, rvc_model_worker, vc_worker, net_g_worker, version_worker, tgt_sr_worker, shm_worker

            # Set GPU device for the worker if gpu_id is provided and CUDA is available
            (shm_name, shm_shape, shm_dtype, input_path, times, pitch_change, f0_method, index_path,
             index_rate, if_f0, filter_radius, rms_mix_rate, protect, crepe_hop_length, index, crossfade_length, gpu_id) = chunk_info_with_gpu # Added gpu_id

            if gpu_id is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_device(gpu_id)
                    # print(f"Worker process {os.getpid()} using GPU {gpu_id}") # Optional: for debugging
                except Exception as e:
                    print(f"Warning: Could not set GPU {gpu_id} for worker {os.getpid()}: {e}")
                    # Continue with default device or handle error


            if hubert_model_worker is None or net_g_worker is None:
                 print(f"Models not loaded in worker {os.getpid()}. Cannot process chunk.")
                 return (index, Exception("Models not loaded in worker"))


            try:
                # Attach to the shared memory and reconstruct the numpy array
                existing_shm = shared_memory.SharedMemory(name=shm_name)
                audio_chunk = np.ndarray(shm_shape, dtype=shm_dtype, buffer=existing_shm.buf)

                # Perform VC inference using the pre-loaded models
                # Ensure vc_worker uses the correct device (should be set by torch.cuda.set_device)
                result = vc_worker.pipeline(
                    hubert_model_worker, net_g_worker, 0, audio_chunk, input_path, times, pitch_change,
                    f0_method, index_path, index_rate, if_f0, filter_radius, tgt_sr_worker,
                    0, rms_mix_rate, version_worker, protect, crepe_hop_length
                )

                # Explicitly close the shared memory handle in the worker
                existing_shm.close()

                # Return result with its original index
                return (index, result)

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"WORKER TASK ERROR in process {os.getpid()}: {e}")
                return (index, e)


        # Update chunk_tasks data to include gpu_id
        chunk_tasks_with_gpu = []
        for i, chunk in enumerate(chunks):
             shm = shared_memory.SharedMemory(create=True, size=chunk.nbytes)
             chunk_np_array = np.ndarray(chunk.shape, dtype=chunk.dtype, buffer=shm.buf)
             chunk_np_array[:] = chunk[:]
             gpu_id = i % torch.cuda.device_count() if torch.cuda.is_available() else None # Assign GPU based on chunk index
             chunk_tasks_with_gpu.append((shm.name, chunk.shape, chunk.dtype, input_path, times, pitch_change, f0_method, index_path,
                                  index_rate, if_f0, filter_radius, rms_mix_rate, protect, crepe_hop_length, i, crossfade_length, gpu_id))
             shm_list.append(shm) # Keep track of shm objects


        # Prepare model paths and config for the initializer
        model_paths_for_initializer = {
            'hubert': os.path.join(os.getcwd(), 'infers', 'hubert_base.pt'),
            'rvc': rvc_model_input
        }
        config_dict_for_initializer = {
            'device': vc.device, # Pass the original device string/object
            'is_half': vc.is_half
        }

        num_chunks = len(chunks)
        num_processes = min(cpu_count(), num_chunks)
        if torch.cuda.is_available():
            num_processes = min(torch.cuda.device_count(), num_processes) # Limit processes by GPU count if using GPU

        # Use multiprocessing Pool with initializer
        with Pool(processes=num_processes, initializer=worker_initializer_reverted, initargs=((model_paths_for_initializer, config_dict_for_initializer),)) as pool:
            # Map chunk tasks to worker processes
            # Pass the updated chunk tasks including gpu_id
            processed_results = pool.map(worker_task_with_gpu, chunk_tasks_with_gpu)


        # Collect and sort results and unlink shared memory
        processed_chunks_with_index = []
        for result_tuple in processed_results:
            if isinstance(result_tuple, tuple) and len(result_tuple) == 2:
                 index, result = result_tuple
                 if isinstance(result, Exception):
                     # Handle exception from worker task
                     # Terminate all processes if an error occurs
                     pool.terminate() # Use pool.terminate() for better control
                     pool.join()
                     # Unlink all created shared memory blocks before raising
                     for shm in shm_list:
                         try:
                            shm.close()
                            shm.unlink()
                         except FileNotFoundError:
                             pass # Handle case where shm might already be unlinked

                     raise result

                 processed_chunks_with_index.append((index, result))
            else:
                 print(f"Warning: Unexpected result format from worker: {result_tuple}")

        # Unlink all created shared memory blocks after processing
        for shm in shm_list:
             try:
                shm.close()
                shm.unlink()
             except FileNotFoundError:
                 pass # Handle case where shm might already be unlinked


        # Sort chunks by index
        processed_chunks_with_index.sort(key=lambda x: x[0])

        # Merge chunks with crossfade
        audio_opt = processed_chunks_with_index[0][1]
        for i in range(1, num_chunks):
            current_chunk_result = processed_chunks_with_index[i][1]
            audio_opt = crossfade(audio_opt, current_chunk_result, crossfade_length)


    else:
        # Audio is shorter than chunk_length_sec, process without parallelization
        print(f"Audio is shorter than {chunk_length_sec} seconds. Processing sequentially.")
        # Ensure sequential path also handles device correctly if running on GPU
        audio = audio.copy() # Ensure audio is writable if modified in pipeline

        # Assuming vc and hubert_model are already on the correct device in the main process
        # If not, they might need to be moved:
        # vc.device = some_device
        # hubert_model.to(some_device)
        # net_g.to(some_device) # net_g is part of vc but might need explicit handling

        audio_opt = vc.pipeline(
            hubert_model, net_g, 0, audio, input_path, times, pitch_change, f0_method,
            index_path, index_rate, if_f0, filter_radius, tgt_sr, 0, rms_mix_rate, version,
            protect, crepe_hop_length
        )

    wavfile.write(output_path, tgt_sr, audio_opt)
