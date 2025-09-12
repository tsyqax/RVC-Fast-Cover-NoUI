import torch
import librosa
import numpy as np

from torchfcpe import spawn_bundled_infer_model


class FCPE:
    def __init__(self, model_path, is_half, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.is_half = is_half

        self.model = spawn_bundled_infer_model(device=self.device)

        if is_half:
            self.model = self.model.half()
        self.model.eval()

    def infer_from_audio(self, audio, thred=0.006):
        sr = 16000
        hop_size = 160
        
        audio = librosa.to_mono(audio)
        audio_length = len(audio)
        f0_target_length = (audio_length // hop_size) + 1
        
        audio = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(-1).to(self.device)

        f0 = self.model.infer(
            audio,
            sr=sr,
            decoder_mode='local_argmax',
            threshold=thred,
            f0_min=80,
            f0_max=880,
            interp_uv=False,
            output_interp_target_length=f0_target_length,
        )

        return f0.squeeze().cpu().numpy()
