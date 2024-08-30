import librosa 
from transformers import AutoProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq


processor = AutoProcessor.from_pretrained("whisper-tiny")
model = ORTModelForSpeechSeq2Seq.from_pretrained("whisper-tiny")

audio, sr = librosa.load("test10s.wav")
audio_np = librosa.resample(audio, orig_sr=sr, target_sr=16000)

input_features = processor(
    audio_np, sampling_rate=16000, return_tensors="pt"
).input_features

gen_tokens = model.generate(input_features)
outputs = processor.batch_decode(gen_tokens, skip_special_tokens=True)

print(outputs)