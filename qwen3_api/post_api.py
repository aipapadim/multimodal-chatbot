import requests

payload = {
    "message": {
        "text": "Extract all objects and spatial relations.",
        "files": ["/path/to/your/image.jpg"]
    }
}

res = requests.post("http://127.0.0.1:5555/api/relations", json=payload)
print(res.json())

