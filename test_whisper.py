import librosa

import sys
import os

import onnxruntime as ort

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from flash_whisper.onnx import ORTWhisper

audio_path = "./sample0.flac"
audio, sr = librosa.load(audio_path)
audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

# whisper_model = ORTWhisper(encoder=ort.InferenceSession("./whisper-tiny/encoder_model.onnx"),
#                            decoder=ort.InferenceSession("./whisper-tiny/decoder_model_merged.onnx"))

# import time

# start_time = time.time()
# temp = whisper_model(audio, sampling_rate=16000)
# print(f"using time: {(time.time() - start_time) * 1000} ms")
# print(temp)

whisper_model = ORTWhisper(encoder=ort.InferenceSession("./whisper-tiny-with-past/encoder_model.onnx"),
                           decoder=ort.InferenceSession("./whisper-tiny-with-past/decoder_model.onnx"),
                           decoder_with_past=ort.InferenceSession("./whisper-tiny-with-past/decoder_with_past_model.onnx"))

import time

start_time = time.time()
temp = whisper_model(audio, sampling_rate=16000)
print(f"using time: {(time.time() - start_time) * 1000} ms")
print(temp)