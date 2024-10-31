import os
import torch
import numpy as np
import torch.nn.functional as F

from typing import Union

def mel_filters(device, n_mels:int = 128) -> torch.Tensor:
    
    assert n_mels == 80 or n_mels == 128, f"Unsupported n_mels: {n_mels}"
    
    with np.load(
        os.path.join(os.path.dirname(__file__), "mel_filters.npz")
    ) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)
    
def log_mel_spectrogram(
    audio: Union[torch.Tensor],
    filters:torch.Tensor,
    n_fft:int = 400,
    hop_length:int = 160,
):
    window = torch.hann_window(n_fft).to(audio.device)
    stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    
    log_spec = log_spec.half()
    return log_spec

class FeatureExtractor(torch.nn.Module):
    
    def __init__(self, n_mels:int = 128):
        self.device = torch.device("cuda")
        self.n_mels = n_mels
        self.filters = mel_filters(self.device, n_mels=self.n_mels)
        
    def compute_feature(self, wav, padding_target_len:int = 3000):
        mel = log_mel_spectrogram(wav, self.filters)
        assert padding_target_len <= 3000, f"padding must be less than 3000, got {padding_target_len}"
        if mel.shape[1] < padding_target_len:
            mel = F.pad(mel, (0, padding_target_len - mel.shape[1]), mode='constant')
        if mel.shape[1] % 2:
            mel = F.pad(mel, (0, 1))
            
        mel = mel.unsqueeze(0)
        return mel