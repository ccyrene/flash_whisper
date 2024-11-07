import requests
import base64
import pickle

url = "http://localhost:8080/transcribe"

# path = "sample0.flac"
# path = "test10s.wav"
path = "test6min30s.wav"
# path = "stereo.wav"

with open(path, "rb") as file:
    bpayload = file.read()
    
response = requests.post(
        url,
        json={
            "audio": base64.encodebytes(bpayload).decode('utf-8'),
            "language": "th",
            "max_new_tokens": 110,
            "chunk_duration": 5
            # "audio": None
        }
)

if response.status_code == 200:
    print("Request succeeded with status code:", response.status_code)
        # Extracting the "text" from the JSON response
    text = response.json()["text"]
    
    # Write the text to a file
    with open("result.txt", "w") as file:
        file.write(text.replace("\n", "\n"))
else:
    print("Request failed with status code:", response.status_code)
    print(response.text)