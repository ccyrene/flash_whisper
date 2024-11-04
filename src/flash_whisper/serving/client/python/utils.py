import types
import scipy
import numpy as np
import soundfile as sf

from io import BytesIO
from typing import Union, List

import tritonclient
from tritonclient.utils import np_to_triton_dtype

def split_data(data, k):
    n = len(data)
    if n < k:
        print(
            f"Warning: the length of the input list ({n}) is less than k ({k}). Setting k to {n}."
        )
        k = n

    quotient = n // k
    remainder = n % k

    result = []
    start = 0
    for i in range(k):
        if i < remainder:
            end = start + quotient + 1
        else:
            end = start + quotient

        result.append(data[start:end])
        start = end

    return result


def process_large_audio_np(samples, chunk_length=30, target_sr=16000):
    samples_per_chunk = chunk_length * target_sr
    total_samples = len(samples)
    num_chunks = (total_samples + samples_per_chunk - 1) // samples_per_chunk  # ceiling division
    chunks = [samples[i * samples_per_chunk:(i + 1) * samples_per_chunk] for i in range(num_chunks)]

    return chunks

def load_audio(bpayload: bytes, sr: int = 16000):
    waveform, sample_rate = sf.read(BytesIO(bpayload), dtype=np.float32)
    if sample_rate != 16000:
        waveform = scipy.signal.resample(waveform, int(len(waveform)*sr/sample_rate))
    return waveform

def postprocess_string(res: Union[str, List[str]]):
    if isinstance(res, str):
        return res
    elif isinstance(res, list):
        if all(isinstance(item, str) for item in res):
            return " ".join(res)
        else:
            return "\n".join(postprocess_string(sublist) for sublist in res)

async def send_whisper(
    dps: list,
    name: str,
    triton_client: tritonclient.grpc.aio.InferenceServerClient,
    protocol_client: types.ModuleType,
    model_name: str,
    max_new_tokens: int = 128,
    whisper_prompt: str = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
):
    results = []
    task_id = int(name[5:])
    for i, dp in enumerate(dps):

        samples = np.zeros(
            (
                1,
                480000
            ),
            dtype=np.float32,
        )

        samples[0, : len(dp)] = dp

        lengths = np.array([[len(dp)]], dtype=np.int32)

        inputs = [
            protocol_client.InferInput(
                "WAV", samples.shape, np_to_triton_dtype(samples.dtype)
            ),
            protocol_client.InferInput(
                "WAV_LENS", lengths.shape, np_to_triton_dtype(lengths.dtype)
            ),
            protocol_client.InferInput("TEXT_PREFIX", [1, 1], "BYTES"),
            protocol_client.InferInput("MODEL_NAME", [1, 1], "BYTES"),
            protocol_client.InferInput("MAX_NEW_TOKENS", [1, 1], "INT32"),
        ]
        
        inputs[0].set_data_from_numpy(samples)
        inputs[1].set_data_from_numpy(lengths)

        input_data_numpy = np.array([whisper_prompt], dtype=object)
        input_data_numpy = input_data_numpy.reshape((1, 1))
        inputs[2].set_data_from_numpy(input_data_numpy)
        
        input_data_numpy = np.array([model_name], dtype=object)
        input_data_numpy = input_data_numpy.reshape((1, 1))
        inputs[3].set_data_from_numpy(input_data_numpy)
        
        input_data_numpy = np.array([max_new_tokens], dtype=np.int32)
        input_data_numpy = input_data_numpy.reshape((1, 1))
        inputs[4].set_data_from_numpy(input_data_numpy)

        outputs = [protocol_client.InferRequestedOutput("TRANSCRIPTS")]
        sequence_id = 100000000 + i + task_id * 10
        response = await triton_client.infer(
            "infer_bls", inputs, request_id=str(sequence_id), outputs=outputs
        )

        decoding_results = response.as_numpy("TRANSCRIPTS")[0]
        if type(decoding_results) == np.ndarray:
            decoding_results = b" ".join(decoding_results).decode("utf-8")
        else:
            decoding_results = decoding_results.decode("utf-8")

        results.append(decoding_results.split())

    return results