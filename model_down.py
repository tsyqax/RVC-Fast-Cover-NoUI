#model_down.py
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

def decrypt_text(encoded_str, shift=-3):
    lower_from = "abcdefghijklmnopqrstuvwxyz"
    lower_to   = "xyzabcdefghijklmnopqrstuvw"
    upper_from = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    upper_to   = "XYZABCDEFGHIJKLMNOPQRSTUVW"
    tab = str.maketrans(lower_from + upper_from, lower_to + upper_to)
    return encoded_str.translate(tab)

os.makedirs('assets/hubert_base', exist_ok=True)
os.makedirs('assets/rmvpe', exist_ok=True)
os.makedirs('assets/mdx', exist_ok=True)

print('MODEL DOWNLOAD STARTED...')


encoded_repo = "om1995/YrlfhFrqyhuvlrqZheXL"
clear_repo = decrypt_text(encoded_repo)

encoded_revision = "pdlq"
encoded_pattern = "kxehuw_edvh/*"

clear_revision = decrypt_text(encoded_revision)
clear_pattern = decrypt_text(encoded_pattern)

encoded_filename = "upysh.sw"
clear_filename = decrypt_text(encoded_filename)

snapshot_download(
    repo_id=clear_repo,
    revision=clear_revision,
    allow_patterns=clear_pattern,
    local_dir="assets"
)

hf_hub_download(
    repo_id=clear_repo,
    filename=clear_filename,
    revision=clear_revision,
    local_dir="assets/rmvpe"
)

encoded_uvr_repo = "Srolwuhhv/XYU_uhvrxufhv"
clear_uvr_repo = decrypt_text(encoded_uvr_repo)

encoded_voc_ft = "prghov/PGAQhw/XYU-PGA-QHW-Yrf_IW.rqqa"
encoded_karaoke2 = "prghov/PGAQhw/XYU_PGAQHW_NDUD_2.rqqa"

clear_voc_ft = decrypt_text(encoded_voc_ft)
clear_karaoke2 = decrypt_text(encoded_karaoke2)

hf_hub_download(
    repo_id=clear_uvr_repo,
    filename=clear_voc_ft,
    revision=clear_revision,
    local_dir="assets/mdx"
)

hf_hub_download(
    repo_id=clear_uvr_repo,
    filename=clear_karaoke2,
    revision=clear_revision,
    local_dir="assets/mdx"
)

print('MODEL DONE!')
