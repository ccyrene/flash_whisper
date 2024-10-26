import librosa

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from flash_whisper.tllm import WhisperTRTLLM

audio_path = "./sample0.flac"
audio, sr = librosa.load(audio_path)
audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)


model = WhisperTRTLLM("/workspace/assets/whisper-medium/whisper_medium", n_mels=80)

import time

for _ in range(3):
    start_time = time.time()
    temp = model([audio], [audio.shape[0]], "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>")
    print(f"using time: {(time.time() - start_time) * 1000} ms")
    print(temp)