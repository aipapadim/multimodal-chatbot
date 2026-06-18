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

### Running the app

```sh
chmod +x start_pillar_chatbot.sh
bash start_pillar_chatbot.sh
```

## Communication
All servers communicate via REST API using JSON format. The data transferred include images, segmentation masks (RLE-encoded), and object bounding boxes to determine spatial relationships.

## Future Improvements
- Integration of the **DeepSeek** model as an alternative backbone.
- Experimenting with **Diffusion models** for text-to-image generation.
- Refactoring image transfer using **Base64 encoding** instead of paths.

## License
This project is part of the **PILLAR-Robots** initiative and follows the licensing agreements set by the project consortium.