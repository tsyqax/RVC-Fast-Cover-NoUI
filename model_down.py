import os
import requests
from pathlib import Path

def decrypt_text(encoded_str, shift=-3):
    lower_from = "abcdefghijklmnopqrstuvwxyz"
    lower_to   = "xyzabcdefghijklmnopqrstuvw"
    upper_from = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    upper_to   = "XYZABCDEFGHIJKLMNOPQRSTUVW"
    tab = str.maketrans(lower_from + upper_from, lower_to + upper_to)
    return encoded_str.translate(tab)

def dl_model(link, model_name, dir_name):
    actual_url = f"{link}{model_name}"
    with requests.get(actual_url) as r:
        r.raise_for_status()
        with open(dir_name / model_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

os.makedirs('uvrs', exist_ok=True)
os.makedirs('infers', exist_ok=True)

print('MODEL DOWNLOAD STARTED...')

encoded_base_url = "kwwsv://kxjjlqjidfh.fr/om1995/YrlfhFrqyhuvlrqZheXL/uhvroyh/pdlq/"
clear_base_url = decrypt_text(encoded_base_url)

encoded_kh_base_url = "kwwsv://kxjjlqjidfh.fr/nlqgdkha/yrlfh-frqyhuvlrq/uhvroyh/pdlq/"
clear_kh_base_url = decrypt_text(encoded_kh_base_url)

clear_hubert = decrypt_text("kxehuw_edvh.sw")
clear_rmvpe = decrypt_text("upysh.sw")
clear_fcpe = decrypt_text("ifsh.sw")

dl_model(clear_base_url, clear_hubert, Path('infers'))
dl_model(clear_base_url, clear_rmvpe, Path('infers'))
dl_model(clear_kh_base_url, clear_fcpe, Path('infers'))

print('MODEL DONE!')
