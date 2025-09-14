import torch
import torchfcpe
import numpy as np

class FCPE:
    def __init__(self, is_half, device='cpu'):
        self.device = device
        self.is_half = is_half
        
        self.model = torchfcpe.spawn_bundled_infer_model(device=self.device)
        self.model.eval()
        
        if self.is_half:
            self.model = self.model.half()

    def infer_from_audio(self, audio, sr=16000, hop_length=160):
        audio_tensor = torch.from_numpy(audio).float().to(self.device).unsqueeze(0).unsqueeze(-1)
        
        audio_length = len(audio)
        f0_target_length = (audio_length // hop_length) + 1
        
        # 모델 추론
        with torch.no_grad():
            if self.is_half:
                audio_tensor = audio_tensor.half()
            
            f0 = self.model.infer(
                audio_tensor,
                sr=sr,
                decoder_mode='local_argmax',
                threshold=0.006,
                f0_min=80,
                f0_max=880,
                interp_uv=False,
                output_interp_target_length=f0_target_length,
            )
        
        f0_np = f0.squeeze().cpu().numpy()
        pitch = f0_np
        pitchf = f0_np
        
        return pitch, pitchf
