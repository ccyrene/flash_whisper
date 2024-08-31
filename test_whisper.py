import librosa
import numpy as np

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import onnxruntime as ort
from flash_whisper.onnx import ORTWhisper

audio_path = "./test10s.wav"
audio, sr = librosa.load(audio_path)
audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

encoder_session = ort.InferenceSession("./whisper-tiny/encoder_model.onnx")

whisper_model = ORTWhisper(encoder=encoder_session)

out = whisper_model(audio, sampling_rate=16000, language="en")
print(out)