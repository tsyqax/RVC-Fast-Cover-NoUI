import torch
import librosa
import numpy as np

from torchfcpe import spawn_bundled_infer_model


class FCPE:
    def __init__(self, model_path, is_half, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.is_half = False # Force is_half to False
        
        self.model = spawn_bundled_infer_model(device=self.device)
        self.model.eval()

    def infer_from_audio(self, audio, thred=0.006):
        sr = 16000
        hop_size = 160
        
        # 1. Ensure the audio data is a PyTorch tensor on the correct device
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).to(self.device)
        
        # 2. Use librosa to ensure audio is mono and the correct shape
        #    Librosa requires a numpy array.
        if isinstance(audio, torch.Tensor):
            audio_for_librosa = audio.cpu().numpy()
        else:
            audio_for_librosa = audio
        audio_for_librosa = librosa.to_mono(audio_for_librosa)

        # 3. Convert back to PyTorch tensor for model inference.
        audio = torch.from_numpy(audio_for_librosa).to(self.device).float()
        
        # Add batch and channel dimensions
        audio = audio.unsqueeze(0).unsqueeze(-1)
        
        audio_length = audio.shape[2]
        f0_target_length = (audio_length // hop_size) + 1
        
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
