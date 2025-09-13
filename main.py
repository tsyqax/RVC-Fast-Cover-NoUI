#main.py
import argparse
import glob
import json
import os
import sys
import shutil
from pydub import AudioSegment
from scipy.io import wavfile # Import wavfile to read WAV data explicitly
import numpy as np # Import numpy

# sys.path.append(os.path.dirname(__file__)) # Assuming my_utils and vc_infer_pipeline are in the same directory as this script
# Adjusting path for Colab environment if necessary, or rely on sys.path additions made earlier
# from my_utils import load_audio # Already imported in rvc.py and vc_infer_pipeline.py
# from vc_infer_pipeline import VC # Already imported in rvc.py

# Assuming rvc.py and vc_infer_pipeline.py are in the same directory structure or paths are handled
from rvc import rvc_infer, Config, load_hubert, get_vc # Import necessary functions from rvc.py


def rvc_song(rvc_index_path, rvc_model_path, index_rate, input_path, output_path, pitch_change, f0_method, filter_radius, rms_mix_rate, protect, crepe_hop_length):
    print(f"Starting RVC inference for {input_path}")

    # Load models and config in the main process
    # This will be used for sequential processing (<60s) or passed to the initializer/task for parallel
    config = Config(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), torch.cuda.is_available()) # Assume CUDA if available
    cpt, version, net_g, tgt_sr, vc = get_vc(config.device, config.is_half, config, rvc_model_path)
    hubert_model = load_hubert(config.device, config.is_half, os.path.join(BASE_DIR, 'DIR', 'infers', 'hubert_base.pt'))


    # rvc_infer function now handles both sequential and parallel logic internally
    rvc_infer(rvc_index_path, index_rate, input_path, output_path, pitch_change, f0_method, cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model, rvc_model_path)

    print(f"RVC inference finished. Output saved to {output_path}")


def split_song(audio_path, output_dir, vocal_split_model='htdemucs', segment_duration=10):
    print(f"Splitting song: {audio_path} using {vocal_split_model}")
    # This function remains largely the same, using the installed splitter (e.g., Demucs)
    # Ensure Demucs or other splitter is installed and configured to output to output_dir
    # Placeholder for actual splitting command
    # Example: !demucs --split-vocals --two-stems=vocals "{audio_path}" -o "{output_dir}"

    # Create dummy files for demonstration if actual splitting is not set up
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dummy_vocal_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}_vocals.wav")
    dummy_other_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}_other.wav")

    if not os.path.exists(dummy_vocal_path):
        # Create dummy vocal and other files (silent or simple tone)
        try:
            original_audio = AudioSegment.from_file(audio_path)
            original_audio.export(dummy_vocal_path, format="wav") # Create a copy as dummy vocal
            AudioSegment.silent(duration=len(original_audio)).export(dummy_other_path, format="wav") # Create dummy silent other
            print("Created dummy vocal and other files.")
        except Exception as e:
             print(f"Could not create dummy audio files: {e}. Ensure pydub dependencies are met.")
             # Create empty files if audio processing fails
             if not os.path.exists(dummy_vocal_path): open(dummy_vocal_path, 'a').close()
             if not os.path.exists(dummy_other_path): open(dummy_other_path, 'a').close()


    return dummy_vocal_path, dummy_other_path


def merge_song(song_name, song_id, rvc_name, vocal_sound_path, other_sound_path, sep_mode):
    print(f"Merging song: {song_name} (ID: {song_id})")

    output_dir = f"./output/{song_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, f"{song_name}_merged_{rvc_name}.wav")

    try:
        # Read vocal and other tracks using wavfile.read for better control over format
        # This assumes the WAV files are standard PCM formats
        rate_vocal, vocal_data = wavfile.read(vocal_sound_path)
        rate_other, other_data = wavfile.read(other_sound_path)

        # Convert numpy arrays to pydub AudioSegment
        # Need to specify sample_width and frame_rate explicitly
        # wavfile.read typically returns int16 for 16-bit WAVs, so sample_width is 2
        # Adjust sample_width if your audio is 24-bit (3) or 32-bit float (4) etc.
        sample_width_vocal = vocal_data.dtype.itemsize
        sample_width_other = other_data.dtype.itemsize

        if vocal_data.ndim > 1: # Handle stereo if necessary
            vocal_data = vocal_data.T.flatten() # Flatten stereo to mono or handle channels appropriately for pydub
        if other_data.ndim > 1:
             other_data = other_data.T.flatten() # Flatten stereo to mono


        # Create AudioSegment objects
        vocal_audio = AudioSegment(
            vocal_data.tobytes(),
            frame_rate=rate_vocal,
            sample_width=sample_width_vocal,
            channels=1 # Assuming mono after potential flattening
        )
        other_audio = AudioSegment(
            other_data.tobytes(),
            frame_rate=rate_other,
            sample_width=sample_width_other,
            channels=1 # Assuming mono
        )


        # Ensure both audio segments have the same frame rate before merging
        if vocal_audio.frame_rate != other_audio.frame_rate:
             print(f"Warning: Vocal ({vocal_audio.frame_rate} Hz) and other ({other_audio.frame_rate} Hz) sample rates mismatch. Resampling vocal.")
             # Resample vocal to match other, or vice versa. Resampling vocal to target SR is common.
             # Assuming target SR is vocal_audio.frame_rate for merging with other_audio
             other_audio = other_audio.set_frame_rate(vocal_audio.frame_rate)
             # Or resample both to a common rate like 44100 or tgt_sr if known


        # Ensure both segments have the same length before mixing
        min_len = min(len(vocal_audio), len(other_audio))
        vocal_audio = vocal_audio[:min_len]
        other_audio = other_audio[:min_len]


        # Apply gain - this is where the audioop.error likely occurred before
        # The sample_width should now be correctly inferred or set in AudioSegment creation
        db_gain = 0 # Define db_gain or get from config/args if needed
        vocal_audio = vocal_audio.apply_gain(db_gain)


        # Mix the tracks (simple addition, pydub handles normalization)
        # You might need to adjust levels here
        merged_audio = vocal_audio.overlay(other_audio)


        # Export the merged audio
        merged_audio.export(output_path, format="wav")

        print(f"Song merged successfully to {output_path}")

    except Exception as e:
        print(f"Error during song merging: {e}")
        traceback.print_exc() # Print traceback for merging errors


# Placeholder for main execution flow
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RVC Cover Maker")
    parser.add_argument("--song_id", type=str, required=True, help="Song ID from songs.json")
    parser.add_argument("--rvc_name", type=str, required=True, help="RVC model name")
    parser.add_argument("--index_rate", type=float, default=0.5, help="Index rate")
    parser.add_argument("--rvc_method", type=str, default="rmvpe", help="RVC f0 method (rmvpe, fcpe)")
    parser.add_argument("--rms_rate", type=float, default=0.4, help="RMS mix rate")
    parser.add_argument("--protect", type=float, default=0.4, help="Protect value") # Assuming protect is a float
    parser.add_argument("--crepe_hop_length", type=int, default=128, help="Crepe hop length") # Assuming crepe_hop_length is int
    # Add other arguments as needed

    args = parser.parse_args()

    # Load song data from songs.json
    songs_data = {}
    try:
        with open('songs.json', 'r', encoding='utf-8') as f:
            songs_data = json.load(f)
    except FileNotFoundError:
        print("Error: songs.json not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: Could not decode songs.json. Check file format.")
        sys.exit(1)


    song_info = songs_data.get(args.song_id)
    if not song_info:
        print(f"Error: Song ID '{args.song_id}' not found in songs.json")
        sys.exit(1)

    song_name = song_info.get("name", f"song_{args.song_id}")
    input_path = song_info.get("path")
    sep_mode = song_info.get("sep_mode", "htdemucs") # Default separation mode

    if not input_path or not os.path.exists(input_path):
        print(f"Error: Input song file not found at '{input_path}' for song ID '{args.song_id}'")
        sys.exit(1)

    # Paths for RVC model and index
    # Adjust these paths based on your directory structure
    rvc_model_path = os.path.join(BASE_DIR, 'DIR', 'weights', f"{args.rvc_name}.pth")
    rvc_index_path = os.path.join(BASE_DIR, 'DIR', 'added_ivf_lib', f"{args.rvc_name}.index")

    if not os.path.exists(rvc_model_path):
        print(f"Error: RVC model not found at '{rvc_model_path}'")
        sys.exit(1)

    # Directory for split audio files
    split_output_dir = f"./split/{args.song_id}"
    vocal_path_split = os.path.join(split_output_dir, f"{os.path.splitext(os.path.basename(input_path))[0]}_vocals.wav")
    other_path_split = os.path.join(split_output_dir, f"{os.path.splitext(os.path.basename(input_path))[0]}_other.wav")


    # Step 1: Split song
    # Assuming split_song handles the actual splitting and saves vocal/other to split_output_dir
    # If splitting is done externally, ensure vocal_path_split and other_path_split exist
    if not os.path.exists(vocal_path_split) or not os.path.exists(other_path_split):
         print("Split audio not found, attempting to split...")
         # Call split_song (or external splitter command)
         split_vocal_path, split_other_path = split_song(input_path, split_output_dir, sep_mode)
         vocal_path_split = split_vocal_path
         other_path_split = split_other_path
    else:
         print("Split audio found, skipping splitting.")


    # Step 2: RVC inference on vocal track
    rvc_output_dir = f"./rvc_output/{args.song_id}"
    if not os.path.exists(rvc_output_dir):
        os.makedirs(rvc_output_dir)
    rvc_output_path = os.path.join(rvc_output_dir, f"{os.path.splitext(os.path.basename(vocal_path_split))[0]}_rvc_{args.rvc_name}.wav")


    # Call RVC inference (rvc_infer function in rvc.py)
    # The rvc_infer function now handles loading models and config internally
    # So we don't need to pass cpt, version, net_g, etc directly from here
    # We need to instantiate Config, load models in main process
    # ... (models loaded at the start of rvc_song function, moved there)

    # Call rvc_infer, passing the loaded models and other parameters
    rvc_song(rvc_index_path, rvc_model_path, args.index_rate, vocal_path_split, rvc_output_path, 0, args.rvc_method, 3, args.rms_rate, args.protect, args.crepe_hop_length)


    # Step 3: Merge RVC vocal with other tracks
    # Use the RVC processed vocal track (rvc_output_path) and the original other track (other_path_split)
    # Need to ensure sample rates and lengths match or are handled in merge_song
    # Passing original song_name, song_id, rvc_name for naming output file
    merge_song(song_name, args.song_id, args.rvc_name, rvc_output_path, other_path_split, sep_mode)


    print("Process completed.")

    # Example of how to call from command line:
    # python main.py --song_id <your_song_id> --rvc_name <your_rvc_model_name> --index_rate 0.5 --rvc_method rmvpe --rms_rate 0.4 --protect 0.4 --crepe_hop_length 128
