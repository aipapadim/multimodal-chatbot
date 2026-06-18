import requests

# 1. containers_object, 2. img_path

url = "http://127.0.0.1:4444/api/image-gen"
headers = {
    "Content-Type": "application/json"
}
data = {
    "containers_objects": {'black box': ['spray can'], 'small box': ['tool']},
    "image_path": "/home/dbek/src/LLAMA3.2/application/multimodal/lab_objects/all_45degrees_stand.jpeg"
}

response = requests.post(url, headers=headers, json=data)

json = response.json()

print("Response Body:", json['response'])
