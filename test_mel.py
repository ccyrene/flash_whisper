import os
import numpy as np

with np.load("/workspace/src/flash_whisper/tllm/mel_filters.npz") as f:
    # return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)
    print(f[f"mel_{80}"].shape)
    print(f[f"mel_{128}"].shape)