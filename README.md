# multimodal-chatbot-LLAMA3.2-11B

## Overview
This repository contains the implementation of a multimodal chatbot designed for the PILLAR-Robots project. The chatbot facilitates robot training through interactive dialogue by integrating textual and visual descriptions. The system consists of multiple modules running on separate Flask servers and a core Python script for coordination.

## Project Structure
```
/core
    ... 
    multimodal_chatbot_ui.py
/diffusing
    /stable-diffusion-webui
        ...
        web-ui.sh
/Grounded-SAM-2
   /application
        ...
        /src
            ...
            ai_engine.py
            app.py
            lib.py
            post_api.py
/Inpaint-Anything
    ...
    ai_engine.py
    app.py
    lib.py
    post_api.py
/llama3.2-api
    ...
    /src
        ai_engine.py
        app.py
        lib.py
        post_api.py
```

## Setup
Each module runs in a separate Conda environment. Ensure you have Conda installed before proceeding.

### Downloading Pretrained Weights
Before running the app, create the `pretrained_models` folder inside `Inpaint-Anything` and download the required model weights into it:

```bash
mkdir -p Inpaint-Anything/pretrained_models/big-lama/models
```

| File | Destination | Download Link |
|------|--------------|----------------|
| sam2.1_hiera_base_plus.pt | `Inpaint-Anything/pretrained_models/sam2.1_hiera_base_plus.pt` | https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt |
| sam2.1_hiera_large.pt | `Inpaint-Anything/pretrained_models/sam2.1_hiera_large.pt` | https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt |
| big-lama best.ckpt | `Inpaint-Anything/pretrained_models/big-lama/best.ckpt` **and** `Inpaint-Anything/pretrained_models/big-lama/models/best.ckpt` | https://drive.google.com/drive/folders/1B2x7eQDgecTL0oh3LSIBDGj0fTxs6Ips |

Quick download for the SAM2 weights:
```bash
wget -P Inpaint-Anything/pretrained_models https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
wget -P Inpaint-Anything/pretrained_models https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

For big-lama, the Google Drive link is a folder — download `best.ckpt` manually from there and copy it into **both** destination paths listed above.

### Running the app
```sh
chmod +x start_pillar_chatbot.sh
bash start_pillar_chatbot.sh
```

## Communication
All servers communicate via REST API using JSON format. The data transferred include images, segmentation masks (RLE-encoded), and object bounding boxes to determine spatial relationships.

## License
This project is part of the **PILLAR-Robots** initiative and follows the licensing agreements set by the project consortium.
