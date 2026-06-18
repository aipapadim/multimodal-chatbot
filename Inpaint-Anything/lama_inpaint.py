import os
import sys
import numpy as np
import torch
import yaml
import glob
import argparse
from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.insert(0, str(Path(__file__).resolve().parent / "lama"))
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo

from utils import load_img_to_array, save_array_to_img
from omegaconf import OmegaConf


@torch.no_grad()
def inpaint_img_with_lama(img, mask, config_p, ckpt_p, device="cuda"):
    from omegaconf import OmegaConf
    import torch
    from saicinpainting.training.trainers import load_checkpoint
    from saicinpainting.evaluation.utils import move_to_device

    
    train_config = OmegaConf.load(config_p)
    
    
    predict_config = OmegaConf.create({
        "model": {"path": ckpt_p},
        "dataset": {"img_suffix": ".jpg"}
    })

    
    if 'visualizer' not in train_config:
        train_config.visualizer = OmegaConf.create({'kind': 'noop'})
    if 'trainer' not in train_config:
        train_config.trainer = OmegaConf.create({'kwargs': {'accelerator': None}})
    
    
    train_config.training_model.predict_only = True

    
    print(f"Loading LaMa checkpoint from: {ckpt_p}")
    model = load_checkpoint(train_config, ckpt_p, strict=False, map_location='cpu')
    model.freeze()
    model.to(device)

    orig_height, orig_width = img.shape[:2]
    pad_h = (8 - orig_height % 8) % 8
    pad_w = (8 - orig_width % 8) % 8
    
    
    if pad_h > 0 or pad_w > 0:
        img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        mask_padded = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    else:
        img_padded = img
        mask_padded = mask

    
    img_tensor = torch.from_numpy(img_padded).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    mask_tensor = torch.from_numpy(mask_padded).float().unsqueeze(0).unsqueeze(0)
    
    
    mask_tensor = (mask_tensor > 0).float() 
    
    batch = {
        'image': img_tensor.to(device),
        'mask': mask_tensor.to(device)
    }

    print(f"LaMa is processing {torch.sum(batch['mask']).item()} mask pixels...")
    with torch.no_grad():
        output = model(batch)
        
        res = output.get('inpainted', output.get('predicted_image'))
        
        cur_res = res[0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        cur_res = cur_res[:orig_height, :orig_width]

    return cur_res

def build_lama_model(        
        config_p: str,
        ckpt_p: str,
        device="cuda"
):
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    device = torch.device(device)

    train_config_path = os.path.join(
        predict_config.model.path, 'config.yaml')

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(
        predict_config.model.path, 'models',
        predict_config.model.checkpoint
    )
    model = load_checkpoint(train_config, checkpoint_path, strict=False)
    model.to(device)
    model.freeze()
    return model


@torch.no_grad()
def inpaint_img_with_builded_lama(
        model,
        img: np.ndarray,
        mask: np.ndarray,
        config_p=None,
        mod=8,
        device="cuda"
):
    assert len(mask.shape) == 2
    if np.max(mask) == 1:
        mask = mask * 255
    img = torch.from_numpy(img).float().div(255.)
    mask = torch.from_numpy(mask).float()

    batch = {}
    batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
    batch['mask'] = mask[None, None]
    unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
    batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
    batch = move_to_device(batch, device)
    batch['mask'] = (batch['mask'] > 0) * 1

    batch = model(batch)
    cur_res = batch["inpainted"][0].permute(1, 2, 0)
    cur_res = cur_res.detach().cpu().numpy()

    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]

    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    return cur_res



def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--input_mask_glob", type=str, required=True,
        help="Glob to input masks",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=True,
        help="The path to the lama checkpoint.",
    )


if __name__ == "__main__":
    """Example usage:
    python lama_inpaint.py \
        --input_img FA_demo/FA1_dog.png \
        --input_mask_glob "results/FA1_dog/mask*.png" \
        --output_dir results \
        --lama_config lama/configs/prediction/default.yaml \
        --lama_ckpt big-lama 
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_stem = Path(args.input_img).stem
    mask_ps = sorted(glob.glob(args.input_mask_glob))
    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    img = load_img_to_array(args.input_img)
    for mask_p in mask_ps:
        mask = load_img_to_array(mask_p)
        img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_p).name}"
        img_inpainted = inpaint_img_with_lama(
            img, mask, args.lama_config, args.lama_ckpt, device=device)
        save_array_to_img(img_inpainted, img_inpainted_p)
