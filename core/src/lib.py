import os
import json

def save_history(history, filename):
    filepath = os.path.join('../history', filename)
    with open(filepath, "w") as f:
        json.dump(history, f)

def get_image_sequence(history, message):
    image_sequence = []

    for entry in history:
        if isinstance(entry, dict):
            content = entry.get('content')
            if isinstance(content, tuple):
                image_sequence.extend(content)
            elif isinstance(content, str) and content.endswith(('.jpg','.jpeg','.png','.webp')):
                image_sequence.append(content)

    if message.get('files'):
        image_sequence.extend(message['files'])

    return image_sequence
