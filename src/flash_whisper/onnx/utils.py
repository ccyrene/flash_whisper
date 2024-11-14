import os
import numpy as np
import onnxruntime as ort

from glob import glob
from typing import Dict

def ort_type_to_dtype(dtype):

    if dtype == "tensor(float)":
        expected_dtype = np.float32
    elif dtype == "tensor(double)":
        expected_dtype = np.float64
    elif dtype == "tensor(int32)":
        expected_dtype = np.int32
    elif dtype == "tensor(int64)":
        expected_dtype = np.int64
    elif dtype == "tensor(uint8)":
        expected_dtype = np.uint8
    elif dtype == "tensor(bool)":
        expected_dtype = np.bool_

    return expected_dtype


def initialize_model(model_dir:str) -> Dict:
    model_file = glob(os.path.join(model_dir, "*.onnx"))
    model_map = {os.path.basename(mf): ort.InferenceSession(mf) for mf in model_file}
    
    return model_map
