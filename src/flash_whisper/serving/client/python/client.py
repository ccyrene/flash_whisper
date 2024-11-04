import os
import base64
import uvicorn
import logging
import asyncio
import argparse

from typing import Union
from dotenv import load_dotenv

import tritonclient.grpc.aio as grpcclient

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from utils import load_audio, split_data, process_large_audio_np, postprocess_string, send_whisper

load_dotenv()
app = FastAPI()

triton_client = grpcclient.InferenceServerClient(os.environ["TRITON_SERVER_ENDPOINT"], verbose=False)
protocol_client = grpcclient

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Grpc port of the triton server, default is 8001",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="spawn n process for client",
    )

    return parser.parse_args()

def validate_request(payload: Union[bytes, str, int], param_name:str):
    
    if param_name == "audio":
        if not isinstance(payload, bytes):
            raise HTTPException(detail={"error": f"Invalid '{param_name}' parameter. Must be bytes."}, status_code=400)
    
    if param_name == "model_name":
        if not isinstance(payload, str):
            raise HTTPException(detail={"error": f"Invalid '{param_name}' parameter. Must be string."}, status_code=400)
    
    if param_name == "language":
        if not isinstance(payload, str):
            raise HTTPException(detail={"error": f"Invalid '{param_name}' parameter. Must be string."}, status_code=400)
    
    if param_name == "chunk_duration":
        if not isinstance(payload, int):
            raise HTTPException(detail={"error": f"Invalid '{param_name}' parameter. Must be integer."}, status_code=400)
    
        payload = max(5, min(payload, 30))
    
    if param_name == "max_new_tokens":
        if not isinstance(payload, int):
            raise HTTPException(detail={"error": f"Invalid '{param_name}' parameter. Must be integer."}, status_code=400)
    
        payload = max(2, min(payload, 434))

    if param_name == "num_tasks":
        if not isinstance(payload, int):
            raise HTTPException(detail={"error": f"Invalid '{param_name}' parameter. Must be integer."}, status_code=400)
        
        payload = max(1, payload)
    
    return payload

def set_default(param_name:str):
    
    default_params = {
        "model_name": "whisper_medium",
        "language": "en",
        "chunk_duration": 30,
        "max_new_tokens": 96,
        "num_tasks": 50
    }
    
    return default_params[param_name]

def get_data(request):
    
    parameters = ["audio", "model_name", "language", "chunk_duration", "max_new_tokens", "num_tasks"]
    
    user_inputs = {}
    for param in parameters:
        payload = request.get(param)
        
        if param == "audio":
            payload = base64.b64decode(payload)
        
        if payload is None and param != "audio":
            payload = set_default(param)
            
        validate_request(payload, param)
        
        user_inputs[param] = payload
        
    return user_inputs

@app.get("/")
def healthcheck() -> bool:
    print("I'm still alive!")
    return True

@app.post("/transcribe")
async def main(request: Request):
    
    json_payload = await request.json()
    
    payload = get_data(json_payload)
    audio = load_audio(payload["audio"])
    dps = process_large_audio_np(audio, chunk_length=payload["chunk_duration"])
    dps_list = split_data(dps, payload["num_tasks"])
    num_tasks = min(payload["num_tasks"], len(dps_list))
    
    inference_kwarg = {
        "triton_client": triton_client,
        "protocol_client": protocol_client,
        "model_name": payload["model_name"],
        "max_new_tokens": payload["max_new_tokens"],
        "whisper_prompt": f"<|startoftranscript|><|{payload['language']}|><|transcribe|><|notimestamps|>"
    }
    
    tasks = []
    for i in range(num_tasks):
        task = asyncio.create_task(
                    send_whisper(
                        dps=dps_list[i],
                        name=f"task-{i}",
                        **inference_kwarg
                    )
                )
        
        tasks.append(task)
        
    ans_list = await asyncio.gather(*tasks)
    res = postprocess_string(ans_list)
    
    return JSONResponse(content={"text": res}, status_code=200)

if __name__ == "__main__":

    args = get_args()
    logging.basicConfig()
    uvicorn.run("client:app", host="0.0.0.0", port=args.port, loop="uvloop", log_level="info", workers=args.workers)