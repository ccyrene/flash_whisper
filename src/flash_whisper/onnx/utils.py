import numpy as np

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
