import os
import requests
from pathlib import Path
# Thanks to SociallyIneptWeeb/AICoverGen

def dl_model(link, model_name, dir_name):
    with requests.get(f'{link}{model_name}') as r:
        r.raise_for_status()
        with open(dir_name / model_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

os.makedirs('uvrs', exist_ok=True)
os.makedirs('infers', exist_ok=True)

print('MODEL... MAY 2 MINUTES')
#dl_model('https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/MDX23C_D1581.ckpt', 'MDX23C_D1581.ckpt', '/content/DIR/uvrs')
dl_model('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/', 'hubert_base.pt', Path('/content/DIR/infers'))
dl_model('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/', 'rmvpe.pt', Path('/content/DIR/infers'))
#dl_model('https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/', 'fcpe.pt', Path('/content/DIR/infers'))
