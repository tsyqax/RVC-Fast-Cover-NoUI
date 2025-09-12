#main.py
import torch
import uuid
import subprocess
import argparse
import json
import yt_dlp
import gc
import librosa
import numpy as np
import os
import math


from pydub import AudioSegment

from rvc import Config, load_hubert, get_vc, rvc_infer

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

# input / songname.mp3

# input -> seperate -> pitch -> rvc(vocal_only) -> merge -> output
# keep: seperate only (many time)

# pitch / pitch_inst.mp3
# pitch / pitch_vocal.mp3

# output format = output/ song_id / song_name (rvc model).mp3

def sep_song(song_path, song_name, song_id):
  demucs_command = ["demucs","-d", "cuda", "-n", "mdx", "--mp3", "--two-stems", "vocal", "--segment", "16", song_path]
  subprocess.run(demucs_command, check=True)
  sep_path = os.path.join(os.path.dirname(song_path), 'separated', 'mdx', song_name)
  os.makedirs(sep_path, exist_ok=True)
  if os.listdir(sep_path):
    instis = os.path.join(sep_path, 'accompaniment.mp3') # accompaniment
    vocalis = os.path.join(sep_path, 'vocal.mp3')
    pitch_dir = os.path.join(os.getcwd(), 'pitch')
    os.makedirs(pitch_dir, exist_ok=True)
    keep_dir = os.path.join(os.getcwd(), song_id)
    os.makedirs(keep_dir, exist_ok=True)
    
    subprocess.run(['cp', vocalis, os.path.join(keep_dir, 'sep_vocal.mp3')], check=True)
    subprocess.run(['cp', instis,  os.path.join(keep_dir, 'sep_inst.mp3')], check=True)
    subprocess.run(['mv', vocalis, os.path.join(pitch_dir, 'pitch_vocal.mp3')], check=True)
    subprocess.run(['mv', instis,  os.path.join(pitch_dir, 'pitch_inst.mp3')], check=True)
    #songs = songload()
    songs[song_name] = song_id
    songsave(songs)
    return 1
  else:
    return 0

def pitch_song(pitch_vocal_path, pitch_other_path, pitch_vocal, pitch_other, song_id, song_name, sep_mode):
  try:
    # 삼겹살 = 반키 * 1.2 # (samgyeopsal = semiton * 1.2)
    # 10 삼겹살 = 1 옥타브 # (10 samgyeopsal = 1 octarve)

    def change_pitch(input_file, output_file, pitch_factor):
      filter_string = f"asetrate=44100*{pitch_factor},atempo=1/{pitch_factor}"
      pitch_command = ["ffmpeg", "-i", input_file, "-filter:a", filter_string, "-y", output_file]
      subprocess.run(pitch_command, check=True)
      os.remove(input_file)
    
    if pitch_vocal != 0:
      pitch_vocal = 2 ** (pitch_vocal / 10)
      change_pitch(input_file=pitch_vocal_path, output_file='to_rvc/rvc_vocal.mp3', pitch_factor=pitch_vocal)
    else:
      subprocess.run(['mv', pitch_vocal_path, 'to_rvc/rvc_vocal.mp3'], check=True)
    if sep_mode is True:
      if pitch_other != 0:
        pitch_other = 2 ** (pitch_other / 10)
        change_pitch(input_file=pitch_other_path, output_file='to_merge/mer_inst.mp3', pitch_factor=pitch_other)
      else:
        subprocess.run(['mv', pitch_other_path, 'to_merge/mer_inst.mp3'], check=True)
    print("PITCH..!")

  except Exception as e:
    print(f"PITCH_ERROR: {e}")

def rvc_song(rvc_index_path, rvc_model_path, index_rate, input_path, output_path, pitch_change, f0_method, filter_radius, rms_mix_rate, protect, crepe_hop_length):
  device = 'cuda:0'
  config = Config(device, True)
  hubert_model = load_hubert(device, config.is_half, os.path.join(os.getcwd(), 'infers', 'hubert_base.pt'))
  cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)

  # convert main vocals
  rvc_infer(rvc_index_path, index_rate, input_path, output_path, pitch_change, f0_method, cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model)
  del hubert_model, cpt
  gc.collect()



def merge_song(song_name, song_id, rvc_name, vocal_vol, inst_vol, sep_mode):
    vocal_path = os.path.join(os.getcwd(), 'to_merge', 'mer_vocal.mp3')
    inst_path = os.path.join(os.getcwd(), 'to_merge', 'mer_inst.mp3')

    mixed_audio = None
    if vocal_vol > 0:
        try:
            vocal = AudioSegment.from_file(vocal_path)
            db_gain = 20 * math.log10(vocal_vol / 100)
            vocal = vocal.apply_gain(db_gain)
            mixed_audio = vocal
        except FileNotFoundError:
            pass

    if inst_vol > 0 and sep_mode is True:
        try:
            inst = AudioSegment.from_file(inst_path)
            db_gain = 20 * math.log10(inst_vol / 100)
            inst = inst.apply_gain(db_gain)
            
            if mixed_audio:
                mixed_audio = mixed_audio.overlay(inst)
            else:
                mixed_audio = inst
        except FileNotFoundError:
            pass

    if not mixed_audio:
        return
    
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f"{song_name} ({rvc_name}).mp3"
    output_path = os.path.join(output_dir, song_id, output_filename)

    mixed_audio.export(output_path, format="mp3")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI RVC COVER', add_help=True)
    parser.add_argument('-in', '--input', type=str, required=True, help='SONG (URL OR DIRECTORY)')
    parser.add_argument('-rvc', '--rvc-name', type=str, required=True, help='RVC MODEL NAME')
    parser.add_argument('-p1', '--pitch-vocal', type=float, default=0, help='VOCAl PITCH CHANGE')
    parser.add_argument('-p2', '--pitch-other', type=float, default=0, help='OTHER PITCH CHANGE')
    parser.add_argument('-sep', '--sep-mode', type=bool, default=True, help='SEPEPERATE ON OFF')
    parser.add_argument('-irate', '--index-rate', type=float, default=0.75, help='INDEX RATE')
    parser.add_argument('-rms', '--rms-rate', type=float, default=0.8, help='RMS RATE')
    parser.add_argument('-algo', '--rvc-method', type=str, default='rmvpe', help='RVC METHOD')
    parser.add_argument('-s1', '--vocal-sound', type=int, default=100, help='VOCAL SOUND')
    parser.add_argument('-s2', '--other-sound', type=int, default=80, help='OTHER SOUND')
    # BooleanOptionalAction
    args = parser.parse_args()

    global sep_mode, exist_check
    sep_mode = args.sep_mode
    exist_check = False
    yt_mode = False

    song_name = '000'
    song_id = '0002'

    pitch_vocal = args.pitch_vocal
    pitch_other = args.pitch_other
    vocal_sound = args.vocal_sound
    other_sound = args.other_sound

    # for keep
    keep_path = os.path.join(os.getcwd(), 'keep')
    os.makedirs(keep_path, exist_ok=True)

    input_dirdir = os.path.join(os.getcwd(), 'input')
    os.makedirs(input_dirdir, exist_ok=True)


    # input (copy)
    if 'https://' in args.input or 'http://' in args.input: # yt
      ydl_opts = {
          'format': 'bestaudio/best',
          'postprocessors': [{
              'key': 'FFmpegExtractAudio',
              'preferredcodec': 'mp3',
              'preferredquality': '128',
          }],
          'ffmpeg_location': 'ffmpeg',
          'outtmpl': '0000.%(ext)s',
      }
      with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(args.input, download=True)
        song_name = info_dict.get('id', 'default000')
      subprocess.run(['mv', '0000.mp3', f'input/{song_name}.mp3'], check=True)
      yt_mode = True

    else: # drive
      song_file = os.path.basename(args.input)
      try:
        song_name = os.path.basename(args.input).split('.')[0]
      except:
        song_name = song_file
      song_ext = os.path.basename(args.input).split('.')[-1]
      subprocess.run(['cp', args.input, f'input/{song_file}'], check=True)
      if song_ext != 'mp3':
        audio = AudioSegment.from_file(f'input/{song_file}')
        audio.export(f"input/{song_name}.mp3", format="mp3", bitrate="128k")

    try:
      song_id = songs[str(song_name)]
      exist_check = True
    except:
      song_id = str(uuid.uuid4()).split('-')[0]
      songs[song_name] = song_id
      songsave(songs)

    input_path0 = os.path.join(os.getcwd(), 'input')
    os.makedirs(input_path0, exist_ok=True)
    input_path = os.path.join(input_path0, f'{song_name}.mp3')


    if sep_mode is True:
      sep_song(input_path, song_name, song_id)
    elif exist_check is False:
      sep_song(input_path, song_name, song_id)
    else:
      print('NO SEPERATE..')

    if pitch_vocal != 0 or pitch_other != 0:
      pitch_path0 = os.path.join(os.getcwd(), 'pitch')
      if exist_check is True: # to pitch move
        os.makedirs(pitch_path0, exist_ok=True)
        subprocess.run(['cp', f'keep/{song_id}/sep_vocal.mp3', 'pitch/pitch_vocal.mp3'], check=True)
        subprocess.run(['cp', f'keep/{song_id}/sep_inst.mp3', 'pitch/pitch_inst.mp3'], check=True)

      pitch_path1 = os.path.join(pitch_path0, f'pitch_vocal.mp3') # vocal
      if sep_mode is False:
        pitch_path2 = pitch_path1
      else:
        pitch_path2 = os.path.join(pitch_path0, f'pitch_inst.mp3') # inst
      pitch_song(pitch_path1, pitch_path2, pitch_vocal, pitch_other, song_id, song_name, sep_mode)

    rvc_index_path = ''
    rvc_vocal_path = ''
    rvc_models_dir0 = os.path.join(os.getcwd(), 'models')
    rvc_models_dir = os.path.join(rvc_models_dir0, args.rvc_name)
    for filename in os.listdir(rvc_models_dir):
      if filename.endswith(".index"):
        rvc_index_path = os.path.join(rvc_models_dir, filename)
        break

    for filename in os.listdir(rvc_models_dir):
      if filename.endswith(".pth"):
        rvc_model_path = os.path.join(rvc_models_dir, filename)
        break

    rvc_output_path0 = os.path.join(os.getcwd(), 'to_merge')
    os.makedirs(rvc_output_path0, exist_ok=True)
    rvc_output_path = os.path.join(rvc_output_path0, 'mer_vocal.mp3')

    rvc_input_path0 = os.path.join(os.getcwd(), 'to_rvc')
    os.makedirs(rvc_input_path0, exist_ok=True)
    rvc_input_path = os.path.join(rvc_input_path0, 'rvc_vocal.mp3')

    rvc_song(rvc_index_path, rvc_model_path, args.index_rate, rvc_input_path, rvc_output_path, 0, args.rvc_method, 3, args.rms_rate, 0.33, 128)
    
    merge_song(song_name, song_id, args.rvc_name, vocal_sound, other_sound, sep_mode)
    
    print('DONE!!')
