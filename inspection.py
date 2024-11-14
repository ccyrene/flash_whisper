import argparse
import onnxruntime as ort

from flash_whisper.onnx import ORTModelBase

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inspection .onnx model")
    parser.add_argument('--onnx', type=str, help="model's path for inspection")
    args = parser.parse_args()
    base_model = ORTModelBase(ort.InferenceSession(args.onnx))
    print(base_model.summary())