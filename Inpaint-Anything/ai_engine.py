import os
import sys
import torch
import numpy as np
import time
import cv2
import gc
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose

from utils.utils import load_img_to_array, save_array_to_img
from lama_inpaint import inpaint_img_with_lama

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAM2_CHECKPOINT = os.path.join(BASE_DIR, "pretrained_models/sam2.1_hiera_base_plus.pt")
LAMA_CONFIG = os.path.join(BASE_DIR, "pretrained_models/big-lama/config.yaml")
LAMA_CKPT = os.path.join(BASE_DIR, "pretrained_models/big-lama/best.ckpt")

# --- GPU CONFIGURATION ---
DEVICE = "cuda:0"

if torch.cuda.is_available():
    try:
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        print("Memory flushed on GPU 0")
    except Exception as e:
        print(f"Could not access GPU 0: {e}")
        DEVICE = "cpu"

# Initializing SAM 2.1
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

initialize(config_path="configs", version_base="1.2")
SAM2_CONFIG = "sam2.1/sam2.1_hiera_b+.yaml"

predictor = SAM2ImagePredictor(build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE))


def dilate_mask(mask, dilate_factor):
    
    if dilate_factor == 0:
        return (mask > 0).astype(np.uint8) * 255
        
    kernel = np.ones((dilate_factor, dilate_factor), np.uint8)
    mask_uint8 = (mask > 0).astype(np.uint8) * 255 
    dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)
    
    return dilated_mask


def sota_inpaint_engine(image_path, bboxes):
    
    try:
        img = load_img_to_array(image_path)
        h, w, _ = img.shape

        max_dim = 1024
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w 
            print(f"Resized original image to {w}x{h} to save VRAM for SAM 2.1")

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictor.set_image(img)
            final_mask = np.zeros((h, w), dtype=bool)

            for bbox in bboxes:
                y1_n, x1_n, y2_n, x2_n = bbox
                x1, y1 = (x1_n / 1000) * w, (y1_n / 1000) * h
                x2, y2 = (x2_n / 1000) * w, (y2_n / 1000) * h

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                
                input_point = np.array([[cx, cy]], dtype=np.float32)
                input_label = np.array([1], dtype=np.int32)
                
                masks, _, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False
                )

                mask_2d = masks.squeeze() > 0.0
                final_mask = np.logical_or(final_mask, mask_2d)

            final_mask_uint8 = (final_mask.astype(np.uint8)) * 255
            dilated = dilate_mask(final_mask_uint8, 25)

        import gc
        gc.collect()
        torch.cuda.empty_cache()

        print("Starting LaMa Inpainting...")
        img_inpainted = inpaint_img_with_lama(img, dilated, LAMA_CONFIG, LAMA_CKPT, device=DEVICE)

        output_dir = Path(BASE_DIR) / "outputs"
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / f"res_{int(time.time())}.png")

        save_array_to_img(img_inpainted, output_path)
        print(f"Success! Saved to: {output_path}")

        return output_path

    except Exception as e:
        print(f"ERROR in Inpaint Engine: {str(e)}")
        raise e

def sota_move_engine(image_path, source_bbox, dest_bbox, position="center"):
    
    try:
        img = load_img_to_array(image_path)
        h, w, _ = img.shape

        # --- DOWNSCALING ---
        max_dim = 1024
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w 

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictor.set_image(img)
            
            # ---1: CUT (SAM 2.1) ---
            sy1_n, sx1_n, sy2_n, sx2_n = source_bbox
            sx1, sy1 = (sx1_n / 1000) * w, (sy1_n / 1000) * h
            sx2, sy2 = (sx2_n / 1000) * w, (sy2_n / 1000) * h
            
            scx = (sx1 + sx2) / 2.0
            scy = (sy1 + sy2) / 2.0
            
            input_point = np.array([[scx, scy]], dtype=np.float32)
            input_label = np.array([1], dtype=np.int32)
            
            print("Cutting object with SAM 2.1...")
            masks, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=False)
            
            source_mask = masks.squeeze() > 0.0
            
            # Cutout
            y_idx, x_idx = np.where(source_mask)
            if len(y_idx) == 0:
                raise ValueError("SAM did not find the object")
                
            crop_y1, crop_y2 = y_idx.min(), y_idx.max()
            crop_x1, crop_x2 = x_idx.min(), x_idx.max()
            
            object_cutout = np.zeros((h, w, 4), dtype=np.uint8)
            object_cutout[..., :3] = img
            object_cutout[..., 3] = (source_mask * 255).astype(np.uint8)
            
            sticker = object_cutout[crop_y1:crop_y2+1, crop_x1:crop_x2+1]
            
            source_mask_uint8 = (source_mask.astype(np.uint8)) * 255
            dilated_mask = dilate_mask(source_mask_uint8, 25)

        gc.collect()
        torch.cuda.empty_cache()

        # --- 2: HEAL (LaMa) ---
        print("Healing original position with LaMa...")
        img_healed = inpaint_img_with_lama(img, dilated_mask, LAMA_CONFIG, LAMA_CKPT, device=DEVICE)

        # --- 3: PASTE (Alpha Blending) ---
        print(f"Pasting object to '{position}' of location...")
        dy1_n, dx1_n, dy2_n, dx2_n = dest_bbox
        dx1, dy1 = (dx1_n / 1000) * w, (dy1_n / 1000) * h
        dx2, dy2 = (dx2_n / 1000) * w, (dy2_n / 1000) * h
        
        dcx = (dx1 + dx2) / 2.0
        dcy = (dy1 + dy2) / 2.0
        
        sticker_h, sticker_w = sticker.shape[:2]
        
        # --- SOTA SPATIAL AWARENESS OFFSET ---
        padding = 10 
        if position == "top":
            dcy = dy1 - (sticker_h / 2.0) - padding
        elif position == "bottom":
            dcy = dy2 + (sticker_h / 2.0) + padding
        elif position == "left":
            dcx = dx1 - (sticker_w / 2.0) - padding
        elif position == "right":
            dcx = dx2 + (sticker_w / 2.0) + padding
        
        
        
        paste_y = int(dcy - sticker_h / 2.0)
        paste_x = int(dcx - sticker_w / 2.0)
        
        img_result = img_healed.copy()
        
        y1, x1 = max(0, paste_y), max(0, paste_x)
        y2, x2 = min(h, paste_y + sticker_h), min(w, paste_x + sticker_w)
        
        sy1, sx1 = max(0, -paste_y), max(0, -paste_x)
        sy2, sx2 = sy1 + (y2 - y1), sx1 + (x2 - x1)
        
        # Alpha blending
        alpha = sticker[sy1:sy2, sx1:sx2, 3] / 255.0
        for c in range(3):
            img_result[y1:y2, x1:x2, c] = (
                alpha * sticker[sy1:sy2, sx1:sx2, c] +
                (1 - alpha) * img_result[y1:y2, x1:x2, c]
            )

        output_dir = Path(BASE_DIR) / "outputs"
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / f"res_moved_{int(time.time())}.png")

        save_array_to_img(img_result, output_path)
        print(f"Move Success! Saved to: {output_path}")

        return output_path

    except Exception as e:
        print(f"ERROR in Move Engine: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e
