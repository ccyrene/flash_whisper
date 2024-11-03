import requests
import base64

url = "http://localhost:8080/transcribe"

with open("test6min30s.wav", "rb") as file:
    bpayload = file.read()
    
response = requests.post(
        url,
        json={
            "audio": base64.encodebytes(bpayload).decode('utf-8'),
            "language": "th",
            "model_name": "infer_bls",
            "max_new_tokens": 96
        }
    )

if response.status_code == 200:
    print("Request succeed with status code:", response.status_code)
    print(response.json()["text"])
else:
    print("Request failed with status code:", response.status_code)
    print(response.text)