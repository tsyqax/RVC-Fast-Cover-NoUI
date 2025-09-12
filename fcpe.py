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
        
        # 1. librosa가 NumPy 배열을 받도록 합니다.
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
            
        # 2. NumPy 배열을 사용하여 librosa를 호출합니다.
        audio = librosa.to_mono(audio)
        
        # 3. 모노 오디오(NumPy 배열)를 다시 PyTorch 텐서로 변환하여 장치로 보냅니다.
        audio = torch.from_numpy(audio).to(self.device)
        
        # 4. is_half 설정에 따라 데이터 타입을 float16 또는 float32로 맞춥니다.
        if self.is_half:
            audio = audio.half()
        else:
            audio = audio.float()
            
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
