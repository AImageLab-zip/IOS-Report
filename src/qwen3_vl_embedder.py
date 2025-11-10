"""qwen3_vl_embedder.py

Simplified script to extract Qwen visual tokens for a collection of intra-oral
images and save them to disk. This script:

- Loads a Qwen3-VL model and its processor
- Resizes images to TARGET_WIDTH x TARGET_HEIGHT (defaults to 512x512)
- Runs the Qwen visual encoder to obtain visual tokens
- Saves the visual token tensors to `_data/iop_qwen_embeddings/<patient>/<view>.pt`

Only this file is changed by the user request.
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


class Qwen3VLImageEmbedder:
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-4B-Instruct", device: str = "auto", dtype: Optional[torch.dtype] = torch.bfloat16):
        print(f"Loading model {model_name} (device={device}, dtype={dtype})")
        # Keep device_map parameter to support 'auto' mapping; this mirrors other code in repo.
        load_kwargs = {"torch_dtype": dtype}
        # device_map accepts 'auto' or device mapping; pass through the provided device
        load_kwargs["device_map"] = device

        # Attn implementation left default to avoid crash in unknown environments
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
        except Exception as e:
            # fallback to no device_map (safer on CPU-only machines)
            print(f"Warning: from_pretrained raised {e}. Retrying without device_map")
            load_kwargs.pop("device_map", None)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)

        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

    @property
    def device(self):
        # Pick the device from model parameters (works with device_map auto-loaded models)
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _prepare_image(self, pil_image: Image.Image, target_width: int, target_height: int) -> dict:
        # Ensure RGB
        img = pil_image.convert("RGB")
        # Resize to target size
        img = img.resize((target_width, target_height), resample=Image.BICUBIC)
        # Use the model's processor to get pixel values and any grid info
        # Some processors expose `image_processor` attribute; call directly if present.
        if hasattr(self.processor, "image_processor"):
            inputs = self.processor.image_processor(images=[img], return_tensors="pt")
        else:
            inputs = self.processor(images=[img], return_tensors="pt")
        return inputs

    @torch.no_grad()
    def get_visual_tokens_from_pil(self, pil_image: Image.Image, target_width: int = 512, target_height: int = 512) -> torch.Tensor:
        inputs = self._prepare_image(pil_image, target_width, target_height)

        # Move tensors to the device where the model lives
        device = self.device
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            raise ValueError("processor did not return 'pixel_values'")

        pixel_values = pixel_values.to(device)

        # grid_thw may or may not be present depending on processor/model
        grid_thw = inputs.get("image_grid_thw")
        if grid_thw is not None:
            grid_thw = grid_thw.to(device)

        # Call the underlying visual encoder. The internal API may return a tuple
        # (final_tokens, deepstack_features...) or a ModelOutput-like object.
        try:
            vision_out = self.model.model.visual(hidden_states=pixel_values, grid_thw=grid_thw)
        except Exception:
            # Some HF model wrappers may expect different keyword names
            vision_out = self.model.model.visual(pixel_values)

        # Normalize the output handling
        # If tuple-like, take the first element as final visual tokens
        if isinstance(vision_out, tuple):
            final = vision_out[0]
        elif hasattr(vision_out, "last_hidden_state"):
            final = vision_out.last_hidden_state
        elif isinstance(vision_out, dict) and "final_embedding" in vision_out:
            final = vision_out["final_embedding"]
        else:
            # Fallback: try to treat it as tensor
            final = torch.as_tensor(vision_out)

        # final expected shape: (batch, seq_len, hidden_dim)
        if final.dim() == 3 and final.shape[0] == 1:
            return final.squeeze(0).cpu()
        else:
            return final.cpu()


def find_views_for_patient(patient_dir: Path):
    """Return a mapping view_name -> file_path for available views in the intraoral-photos folder."""
    views = ["center", "down", "left", "right", "up"]
    exts = ["jpg", "png", "jpeg", "JPG", "PNG", "JPEG"]
    photos_dir = patient_dir / "intraoral-photos"
    found = {}
    if not photos_dir.exists():
        return found

    for v in views:
        for e in exts:
            p = photos_dir / f"{v}.{e}"
            if p.exists():
                found[v] = p
                break
    return found


def main():
    parser = argparse.ArgumentParser(description="Extract Qwen visual tokens for intra-oral photos")
    parser.add_argument("--source_dir", type=str, default="/work/grana_maxillo/IOS-DraftReport/_data/ios-iop-text",
                        help="Root folder containing patient subfolders with 'intraoral-photos' subfolder")
    parser.add_argument("--out_dir", type=str, default="/work/grana_maxillo/IOS-DraftReport/_data/iop_qwen_embeddings",
                        help="Output directory for embeddings")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--device", type=str, default="auto", help="Device map to pass to from_pretrained (e.g. 'auto' or 'cpu')")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"],
                        help="Dtype for model weights")
    parser.add_argument("--target_size", type=int, default=512, help="Target width and height for resized images")
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    embedder = Qwen3VLImageEmbedder(model_name=args.model_name, device=args.device, dtype=dtype)

    src = Path(args.source_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted([p for p in src.iterdir() if p.is_dir()])

    for patient in tqdm(patient_dirs, desc="Patients"):
        views = find_views_for_patient(patient)
        if not views:
            continue

        out_patient_dir = out_root / patient.name
        out_patient_dir.mkdir(parents=True, exist_ok=True)

        # If all five view files are already present, consider this patient processed and skip.
        expected_views_all = ["center", "down", "left", "right", "up"]
        expected_paths = [out_patient_dir / f"{v}.pt" for v in expected_views_all]
        if all(p.exists() for p in expected_paths):
            # Use tqdm.write to avoid breaking the progress bar output
            tqdm.write(f"Skipping {patient.name}: all {len(expected_paths)} embeddings present")
            continue

        for view_name, img_path in views.items():
            try:
                pil = Image.open(img_path)
            except Exception as e:
                print(f"Could not open {img_path}: {e}")
                continue

            tokens = embedder.get_visual_tokens_from_pil(pil, target_width=args.target_size, target_height=args.target_size)

            # Save tensor (seq_len, hidden_dim) as a .pt file per view
            save_path = out_patient_dir / f"{view_name}.pt"
            try:
                torch.save(tokens, save_path)
            except Exception as e:
                print(f"Failed saving {save_path}: {e}")

    print("Done. Embeddings saved to:", out_root)


if __name__ == "__main__":
    main()