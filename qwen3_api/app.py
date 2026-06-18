from flask import Flask, request, jsonify
from flask_cors import CORS
from ai_engine import generate
import json
import os
import time

app = Flask(__name__)
CORS(app)

@app.route("/api/relations", methods=["POST"])
def relations():
    data = request.json
    response = generate(data)

    # --- Auto save output JSON ---
    os.makedirs("outputs", exist_ok=True)  # Folder created automatically
    timestamp = int(time.time())
    filename = f"outputs/relations_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(response, f, indent=2)

    print(f"Saved structured output → {filename}")

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555)

