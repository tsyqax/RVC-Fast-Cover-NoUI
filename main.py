#main.py
import torch
import uuid
import subprocess
import argparse
import json
import gc
import librosa
import soundfile as sf
import numpy as np
import os
import math
import shutil
import onnxruntime as ort
import soundfile as sf

from pydub import AudioSegment
from pytubefix import YouTube
from pytubefix.cli import on_progress
from concurrent.futures import ThreadPoolExecutor

from rvc import Config, load_hubert, get_vc, rvc_infer

try:
    torch.multiprocessing.set_start_method('spawn', force=True)
    print("spawn")
except RuntimeError as e:
    print(f"{e}")

def songload():
  try:
    with open('songs.json', 'r') as file:
      songs_data = json.load(file)
      if not songs_data or isinstance(songs_data, list):
        return {}
      return songs_data

  except FileNotFoundError:
    with open('songs.json', 'w') as file:
      initdata = {}
      json.dump(initdata, file, indent=2)
      return {}

def songsave(data_to_update):
  try:
    with open('songs.json', 'w') as file:
      json.dump(data_to_update, file, indent=2)
  except Exception as e:
     print(f"ERROR: {e}")

songs = songload()

def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

available_prd = ['CUDAExecutionProvider', 'CPUExecutionProvider']
global imload = None

def loadUVR(model_name):
  if imload is not None:
    unloadUVR()
  
  model_path = os.path.join("assets", "mdx", f"{model_name}.onnx")
  if not os.path.exists(model_path):
    print("Model not found, skip")
    return None
  
  global imload
  imload = ort.InferenceSession(model_path, providers=available_prd)
  print(f"[UVR] Load Successfully: {model_name}")
  return imload

def unloadUVR(model_name):
  global imload
  if imload is not None:
    del imload
    imload = None
  
  gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
  elif hasattr(torch, 'mps') and torch.mps.is_available():
    torch.mps.empty_cache()
  
  print("[UVR] Unload Successfully.")

def switchUVR(unload_model_name, load_model_name):
  print(f"[UVR] Model Switch: {unload_model_name} ➔ {load_model_name}")
  unloadUVR()
  loadUVR(load_model_name)

# input / songname.mp3

# input -> seperate -> pitch -> rvc(vocal_only) -> merge -> output
# keep: seperate only (many time)

# pitch / pitch_inst.mp3
# pitch / pitch_vocal.mp3

# output format = output/ song_id / song_name (rvc model).mp3

def sep_song_v2(song_path, vocal_output_path, inst_output_path, chorus_out_path, keep_dir, song_filename, song_id, pitch_other, chorus_mode=0):
  MAIN_SEP_MODEL = "UVR-MDX-NET-Voc_FT"
  CHORUS_SEP_MODEL = "UVR_MDXNET_KARA_2"  

  sep_path = os.path.join(os.getcwd(), 'separated', 'uvr5_mdx')
  os.makedirs(sep_path, exist_ok=True)
  
  uvr_voc_command = ["audio-separator", song_path, "--model_dir", os.path.join("assets", "mdx"), "--model_filename", f"{MAIN_SEP_MODEL}.onnx", "--output_dir", sep_path, "--output_format", "FLAC", "--output_names", '{"Vocals": "vocal_mixed", "Instrumental": "sep_inst"}']
  subprocess.run(uvr_voc_command, check=True)

  instis = os.path.join(sep_path, 'sep_inst.flac')
  vocal_mixed = os.path.join(sep_path, 'vocal_mixed.flac')

  shutil.copy2(instis, keep_dir)
  pitch_song_new(instis, inst_output_path, pitch_other)

  uvr_kara_command = ["audio-separator", vocal_mixed, "--model_dir", os.path.join("assets", "mdx"), "--model_filename", f"{CHORUS_SEP_MODEL}.onnx", "--output_dir", sep_path, "--output_format", "FLAC", "--output_names", '{"Vocals": "sep_main", "Instrumental": "sep_chorus"}']
  subprocess.run(uvr_kara_command, check=True)

  main_vocal = os.path.join(sep_path, 'sep_main.flac')
  chorus_sound = os.path.join(sep_path, 'sep_chorus.flac')
  
  shutil.copy2(main_vocal, keep_dir)
  shutil.copy2(chorus_sound, keep_dir)

  if chorus_mode == 3:
    shutil.move(chorus_sound, vocal_output_path) # merge with infer
    shutil.move(main_vocal, vocal_output_path)
  elif chorus_mode == 2:
    shutil.move(chorus_sound, chorus_out_path) # merge without infer
    shutil.move(main_vocal, vocal_output_path)
  elif chorus_mode == 1:
    os.remove(chorus_sound) # not merge
    shutil.move(main_vocal, vocal_output_path)
  else:
    shutil.move(vocal_mixed, vocal_output_path)
  
  songs[song_name] = song_id
  songsave(songs)
  print(f"[SEP] Separation is Done, Mode: {chorus_mode}")

def pitch_song_new(input_pitch, output_pitch, pitch_sgs):
  # 삼겹살 * 1.2 = 반키 # (samgyeopsal * 1.2 = semiton)
  # 10 삼겹살 = 1 옥타브 # (10 samgyeopsal = 1 octarve)
  if pitch_sys == 0:
    shutil.copy2(input_pitch, output_pitch)
    return
  
  audio_length = librosa.get_duration(path=input_path) #sec
  
  try:
    if audio_length < 600:
      semitones = pitch_sgs * 1.2
      y, sr = librosa.load(input_pitch, sr=None)
      y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
      sf.write(output_pitch, y_shifted, sr)
    else:
      pitch_factor = 2 ** (pitch_sgs / 10)
      filter_string = f"asetrate=44100*{pitch_factor},atempo=1/{pitch_factor}"
      pitch_command = ["ffmpeg", "-i", input_pitch, "-filter:a", filter_string, "-y", output_pitch]
      subprocess.run(pitch_command, check=True)
    print(f"[PITCH] Changing to {pitch_sgs}sgs is done.")
  except Exception as e:
    print(f"[PITCH] Error, due to {e}")


# Refactor 2
def rvc_song(rvc_index_path, rvc_model_path, index_rate, input_path, output_path, pitch_change, f0_method, filter_radius, rms_mix_rate, protect, crepe_hop_length):
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  config = Config(device, True)
  
  hubert_model = load_hubert(device, config.is_half, os.path.join(os.getcwd(), 'infers', 'hubert_base.pt'))
  cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)

  rvc_infer(rvc_index_path, index_rate, input_path, output_path, pitch_change, f0_method, cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model, rvc_model_path)

  del hubert_model, cpt
  gc.collect()

def new_merge_song(vocal_path, inst_path, chorus_path, output_path, vocal_sound, other_sound, sep_mode, chorus_mode):
  mixed_audio = None
  audio_length = min(librosa.get_duration(path=vocal_path), librosa.get_duration(path=inst_path)) # sec
    
  use_chorus = (sep_mode is True and chorus_mode >= 2 and chorus_path and os.path.exists(chorus_path))

  if audio_length < 600:
    if vocal_sound > 0:
      try:
        vocal = AudioSegment.from_file(vocal_path).set_sample_width(2)
        mixed_audio = vocal.apply_gain(20 * math.log10(vocal_sound / 100))
      except FileNotFoundError:
        pass
        
      if other_sound > 0 and sep_mode is True:
        try:
          inst = AudioSegment.from_file(inst_path).set_sample_width(2)
          db_gain = 20 * math.log10(other_sound / 100)
          inst = inst.apply_gain(db_gain)
          
          if mixed_audio:
            mixed_audio = mixed_audio.overlay(inst)
            if use_chorus:
              chorus = AudioSegment.from_file(chorus_path).set_sample_width(2)
              mixed_audio = mixed_audio.overlay(chorus.apply_gain(db_gain))
          else:
            mixed_audio = inst
        except FileNotFoundError:
          pass

      if not mixed_audio:
        return

      mixed_audio.export(output_path, format="mp3")
      print("[MERGE] Done.")

  else:
    try:
      v_vol = vocal_sound / 100
      i_vol = other_sound / 100

      if sep_mode is True and other_sound > 0:
        if use_chorus:
          filter_str = f"[0:a]volume={v_vol}[v]; [1:a]volume={i_vol}[i]; [2:a]volume={i_vol}[c]; [v][i][c]amix=inputs=3:duration=longest"
          merge_command = ["ffmpeg", "-i", vocal_path, "-i", inst_path, "-i", chorus_path, "-filter_complex", filter_str, "-codec:a", "libmp3lame", "-b:a", "192k", "-y", output_path]
        else:
          filter_str = f"[0:a]volume={v_vol}[v]; [1:a]volume={i_vol}[i]; [v][i]amix=inputs=2:duration=longest"
          merge_command = ["ffmpeg", "-i", vocal_path, "-i", inst_path, "-filter_complex", filter_str, "-codec:a", "libmp3lame", "-b:a", "192k", "-y", output_path]
      else:
        filter_str = f"volume={v_vol}"
        merge_command = ["ffmpeg", "-i", vocal_path, "-filter:a", filter_str, "-codec:a", "libmp3lame", "-b:a", "192k", "-y", output_path]

        subprocess.run(merge_command, check=True)
        print("[MERGE] Done.")

    except Exception as e:
      print(f"[MERGE] Error, due to {e}")

# Refactor 260902
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI RVC COVER', add_help=True)
    parser.add_argument('-in', '--input', type=str, required=True, help='SONG (URL OR DIRECTORY)')
    parser.add_argument('-rvc', '--rvc-name', type=str, required=True, help='RVC MODEL NAME')
    parser.add_argument('-p1', '--pitch-vocal', type=float, default=0, help='VOCAl PITCH CHANGE')
    parser.add_argument('-p2', '--pitch-other', type=float, default=0, help='OTHER PITCH CHANGE')
    parser.add_argument('-sep', '--sep-mode', type=str2bool, default=True, help='SEPERATE ON OFF')
    parser.add_argument('-irate', '--index-rate', type=float, default=0.75, help='INDEX RATE')
    parser.add_argument('-rms', '--rms-rate', type=float, default=0.8, help='RMS RATE')
    parser.add_argument('-algo', '--rvc-method', type=str, default='rmvpe', help='RVC METHOD')
    parser.add_argument('-s1', '--vocal-sound', type=int, default=100, help='VOCAL SOUND')
    parser.add_argument('-s2', '--other-sound', type=int, default=80, help='OTHER SOUND')
    parser.add_argument('-chr', '--chorus-mode', type=int, default=1, choices=[0, 1, 2, 3], help='CHOURS SPERATE')
    
    # BooleanOptionalAction
    args = parser.parse_args()

    global sep_mode, exist_check, parrel_mode
    sep_mode = args.sep_mode
    chorus_mode = args.chorus_mode # 0 = no sep, 1 = sep + discard, 2 = sep + merge, 3 = sep + infer
    exist_check = False
    yt_mode = False

    song_name = '000'
    song_ext = 'mp3'
    song_id = 'no_seperate'

    pitch_vocal = args.pitch_vocal
    pitch_other = args.pitch_other
    vocal_sound = args.vocal_sound
    other_sound = args.other_sound

    input_dir = os.path.join(os.getcwd(), 'input')
    merge_dir = os.path.join(os.getcwd(), 'to_merge')
    rvc_dir = os.path.join(os.getcwd(), 'to_rvc')
    output_dir = os.path.join(os.getcwd(), 'output', song_id)
    
    os.makedirs(merge_dir, exist_ok=True)
    os.makedirs(rvc_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)

    rvc_output_path = os.path.join(merge_dir, 'mer_vocal.flac')
    rvc_chorus_path = os.path.join(merge_dir, 'mer_chorus.flac')
    inst_target = os.path.join(merge_dir, 'mer_inst.flac')
    chorus_target = os.path.join(merge_dir, 'mer_chorus.flac')
    vocal_target = os.path.join(rvc_dir, 'rvc_vocal.flac')

    # input (copy)
    if 'https://' in args.input or 'http://' in args.input: # yt
      try:
        song_name = args.input.split('/')[-1]
        if '=' in song_name:
          song_name = song_name.split('=')[-1]
          if '?' in song_name:
            song_name = song_name.split('?')[-2]
        else:
          song_name = song_name
      except:
        song_name = 'default_id'
      
      yt = YouTube(args.input, on_progress_callback=on_progress, client='MWEB')
      ys = yt.streams.get_audio_only()
      ys.download(output_path='input', filename=f'{song_name}.m4a')
      yt_mode = True
      song_ext = 'm4a'

    else: # drive
      song_file = os.path.basename(args.input)
      try:
        song_name = os.path.basename(args.input).split('.')[0]
      except:
        song_name = song_file
      song_ext = os.path.basename(args.input).split('.')[-1]
      shutil.copy2(args.input, f'input/{song_file}')
    song_filename = os.path.basename(args.input).split('.')[0]
    
    try:
      song_id = songs[str(song_name)]
      exist_check = True
    except Exception as e:
      song_id = str(uuid.uuid4()).split('-')[0]
      print(f"NO ID... or ERR: {e}")

    input_path = os.path.join(input_dir, f'{song_name}.{song_ext}')

    if exist_check is True:
      print('SEPERATE PASS..')
    elif sep_mode is False:
      print("SEPERATE FALSE..")
      rvc_output_path = output_path2
      vocal_target = input_path
    else:
      keep_dir = os.path.join(os.getcwd(), 'keep', song_id)
      os.makedirs(keep_dir, exist_ok=True)
      sep_song(input_path, vocal_target, inst_target, chorus_target keep_dir, song_filename, song_id, pitch_other)

    rvc_index_path = ''
    rvc_vocal_path = ''
    rvc_models_dir = os.path.join(os.getcwd(), 'models')
    rvc_models_path = os.path.join(rvc_models_dir, args.rvc_name)
    for filename in os.listdir(rvc_models_path):
      if filename.endswith(".index"):
        rvc_index_path = os.path.join(rvc_models_path, filename)
        break

    for filename in os.listdir(rvc_models_path):
      if filename.endswith(".pth"):
        rvc_model_path = os.path.join(rvc_models_path, filename)
        break
    
    output_path2 = os.path.join(output_dir, f"{song_name} ({args.rvc_name}).mp3")
    rvc_song(rvc_index_path, rvc_model_path, args.index_rate, vocal_target, rvc_output_path, pitch_vocal * 1.2, args.rvc_method, 3, args.rms_rate, 0.33, 128)
    
    if (chorus_mode == 3):
      rvc_song(rvc_index_path, rvc_model_path, args.index_rate, chorus_target, rvc_chorus_path, pitch_vocal * 1.2, args.rvc_method, 3, args.rms_rate, 0.33, 128)
      chorus_target = rvc_chorus_path
    
    new_merge_song(rvc_output_path, inst_target, chorus_target, output_path2, vocal_sound, other_sound, sep_mode, chorus_mode)

    if sep_mode is True:
      songs[song_name] = song_id
      songsave(songs)
    print(f'DONE!!\nSAVED: {output_path2}')
