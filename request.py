import requests
import base64

url = "http://localhost:8080/transcribe"

path = "./sample/sample0.flac"

with open(path, "rb") as file:
    bpayload = file.read()
    
response = requests.post(
        url,
        json={
            "audio": base64.encodebytes(bpayload).decode('utf-8'),
            "language": "th",
            "max_new_tokens": 110,
            "chunk_duration": 30
        }
)

if response.status_code == 200:
    print("Request succeeded with status code:", response.status_code)
    text = response.json()["text"]
    print(text)
else:
    print("Request failed with status code:", response.status_code)
    print(response.text)
    
    