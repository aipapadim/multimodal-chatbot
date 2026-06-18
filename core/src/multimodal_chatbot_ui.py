import os
import time
import requests
import gradio as gr
import json
from lib import save_history, get_image_sequence

# --- CONFIGURATION ---
QWEN_API_URL = "http://127.0.0.1:5555/api/relations"
INPAINT_API_URL = "http://127.0.0.1:4444/api/image-gen"

# ----------------------------------------------------------------------------
# API CALLS
# ----------------------------------------------------------------------------

def qwen_call(payload):
    
    res = requests.post(QWEN_API_URL, json=payload)
    res.raise_for_status()
    return res.json()

def inpaint_call(payload):
    
    res = requests.post(INPAINT_API_URL, json=payload)
    res.raise_for_status()
    return res.json()

# ----------------------------------------------------------------------------
# MAIN LOGIC
# ----------------------------------------------------------------------------

def answer(message, history, history_file, max_new_tokens, temperature, top_p, sys_prompt):

    image_sequence = get_image_sequence(history, message)
    if not image_sequence:
        yield "Please upload an image!"
        return
        
    image_path = image_sequence[-1] 
    
    
    user_text = message.get('text', '')
    user_text_lower = user_text.lower()

    
    edit_keywords = ['remove', 'delete', 'erase', 'svise', 'vgale', 'clean', 'αφαίρεσε', 'σβήσε']
    move_keywords = ['move', 'place', 'put', 'μετάφερε', 'μετακίνησε', 'βάλε', 'πήγαινε']
    
    # ==========================================
    # MODE 1: EDIT
    # ==========================================
    if any(kw in user_text_lower for kw in edit_keywords):
        yield "Object Detection..."
        
        qwen_payload = {
            "message": message,
            "intent": "edit"
        }

        try:
            qwen_res = qwen_call(qwen_payload)
            bboxes = [obj['bbox'] for obj in qwen_res.get('objects', [])]
            
            if not bboxes:
                yield "Did not find the object. Describe it again."
                return

            yield f"Object detected. Continue with editing..."
            
            inpaint_res = inpaint_call({"intent": "edit", "image_path": image_path, "bboxes": bboxes})
            result_path = inpaint_res.get("response")

            yield f"Object Removed."
            yield gr.Image(result_path)
            return 
            
        except Exception as e:
            yield f"Error during processing: {str(e)}"
            return

    # ==========================================
    # MODE 2: MOVE 
    # ==========================================
    elif any(kw in user_text_lower for kw in move_keywords):
        yield "Location analysis"
        
        qwen_payload = {
            "message": message,
            "intent": "move"
        }

        try:
            qwen_res = qwen_call(qwen_payload)
            
            if "error" in qwen_res:
                yield f"Error: {qwen_res['error']}"
                return

            source = qwen_res.get("source")
            destination = qwen_res.get("destination")
            position = qwen_res.get("position", "center") 
            
            if not source or not destination:
                yield "Which object to reposition and where"
                return

            yield f"Reposition: {source['name']} --> {position} of {destination['name']}..."
            
            
            inpaint_payload = {
                "intent": "move",
                "image_path": image_path,
                "source_bbox": source["bbox"],
                "dest_bbox": destination["bbox"],
                "position": position 
            }
            
            inpaint_res = inpaint_call(inpaint_payload)
            result_path = inpaint_res.get("response")

            yield f"Ready"
            yield gr.Image(result_path)
            return
            
        except Exception as e:
            yield f"Error during repositioning: {str(e)}"
            return

    # ==========================================
    # MODE 3: CHAT
    # ==========================================
    yield "Thinking..."
    payload = {
        "message": message,
        "intent": "chat"
    }

    try:
        response_json = qwen_call(payload)
        reply = response_json.get("response", str(response_json))
        
        buffer = ""
        for char in reply:
            buffer += char
            yield buffer
            time.sleep(0.005)
            
    except Exception as e:
        yield f"Error API: {str(e)}"


# ----------------------------------------------------------------------------
# INTERFACE
# ----------------------------------------------------------------------------

demo = gr.ChatInterface(
    fn=answer,
    title="Pillar Multimodal Assistant",
    type="messages",
    additional_inputs=[
        gr.Textbox(label="History file", value="history/chat_log.json"),
        gr.Slider(16, 1024, 512, label="Tokens"),
        gr.Slider(0.1, 2.0, 0.7, label="Temperature"),
        gr.Slider(0.1, 1.0, 0.9, label="Top-P"),
        gr.Textbox(
            label="System Prompt",
            value="You are an Image Editor. Detect objects in [ymin, xmin, ymax, xmax] format (0-1000)."
        ),

        ],
            multimodal=True,
            textbox=gr.MultimodalTextbox(file_types=["image"]),
            fill_height=True,
        )

# Υπολογίζει δυναμικά την απόλυτη διαδρομή για τον φάκελο outputs
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "Inpaint-Anything", "outputs")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        debug=True,
        allowed_paths=[OUTPUTS_DIR]
    )



