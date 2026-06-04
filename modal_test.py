# -*- coding: utf-8 -*-
"""
modal_test.py
RMBG-2.0 Pants Segmentation — Modal GPU Deployment with FastAPI Web Server

Run locally:
    modal serve modal_test.py

Deploy to production:
    modal deploy modal_test.py
"""

import io
import os
import base64
from pathlib import Path

import modal

# ─────────────────────────────────────────────────────────────────────────────
# MODAL IMAGE
# All dependencies baked into the container image.
# ─────────────────────────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Install timm & kornia FIRST in their own layer so they are always
    # present before transformers tries to import them via trust_remote_code.
    .pip_install(
        "timm>=0.9.16",
        "kornia>=0.7.0",
    )
    .pip_install(
        "torch",
        "torchvision",
        "transformers>=4.40.0",
        "Pillow",
        "opencv-python-headless",
        "huggingface_hub",
        "numpy",
        "accelerate",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "einops",      # required by some RMBG-2.0 custom layers
        "safetensors",
    )
    .add_local_file("index.html", remote_path="/root/index.html")
)

# ─────────────────────────────────────────────────────────────────────────────
# MODAL APP
# ─────────────────────────────────────────────────────────────────────────────

app = modal.App("rmbg-pants-segmentation", image=image)

# Persistent volume — model weights are downloaded once and cached here.
model_cache = modal.Volume.from_name("rmbg-model-cache", create_if_missing=True)
CACHE_DIR = "/root/.cache/huggingface"

# ─────────────────────────────────────────────────────────────────────────────
# SECRETS  (set your HF token via: modal secret create hf-secret HF_TOKEN=hf_xxx)
# ─────────────────────────────────────────────────────────────────────────────

hf_secret = modal.Secret.from_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS  (run inside the container)
# ─────────────────────────────────────────────────────────────────────────────

def _load_rmbg():
    from transformers import AutoModelForImageSegmentation
    import torch

    model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-2.0",
        trust_remote_code=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, device


def _load_dino():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    import torch

    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return processor, model, device


def _detect_pants(image, processor, dino, device, text, threshold=0.35):
    import torch

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs.input_ids,
        target_sizes=[image.size[::-1]],
    )
    boxes  = results[0]["boxes"]
    scores = results[0]["scores"]
    valid  = scores > threshold
    return boxes[valid], scores[valid]


def _run_rmbg(image, rmbg_model, device):
    import torch
    import numpy as np
    import cv2
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])
    original      = np.array(image)
    original_size = image.size
    input_tensor  = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = rmbg_model(input_tensor)[-1]

    mask = pred[0].squeeze().cpu().numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask = cv2.resize(mask, original_size)
    mask = (mask * 255).astype(np.uint8)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contour found in segmentation mask.")

    largest    = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, cv2.FILLED)
    clean_mask = cv2.GaussianBlur(clean_mask, (7, 7), 0)

    return original, clean_mask, original_size


def _apply_overlay(original, clean_mask, overlay_color=(255, 105, 180), opacity=0.35, transparent=True):
    import numpy as np
    import cv2
    from PIL import Image

    h, w     = clean_mask.shape
    original = cv2.resize(original, (w, h))
    overlay  = np.zeros_like(original)
    overlay[:] = overlay_color

    alpha    = clean_mask.astype(np.float32) / 255.0
    alpha_3d = np.expand_dims(alpha, axis=2)

    if transparent:
        result = (
            original.astype(np.float32) * (1 - opacity * alpha_3d)
            + overlay.astype(np.float32) * (opacity * alpha_3d)
        ).astype(np.uint8)
        rgba  = np.dstack([result, clean_mask])
        edges = cv2.Canny(clean_mask, 100, 200)
        rgba[edges > 0] = [overlay_color[0], overlay_color[1], overlay_color[2], 255]
        output_arr = rgba
        mode = "RGBA"
    else:
        white_bg   = np.ones((h, w, 3), dtype=np.float32) * 255
        foreground = original.astype(np.float32) * alpha_3d + white_bg * (1 - alpha_3d)
        result     = np.where(
            alpha_3d > 0,
            foreground * (1 - opacity * alpha_3d) + overlay * (opacity * alpha_3d),
            white_bg,
        ).astype(np.uint8)
        edges = cv2.Canny(clean_mask, 100, 200)
        result[edges > 0] = list(overlay_color)
        output_arr = result
        mode = "RGB"

    # ── Orientation fix ──────────────────────────────────────────────────────
    orientation_msg = []

    ys, xs = np.where(clean_mask > 100)
    if len(ys) > 0 and len(xs) > 0:
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        obj_width = x_max - x_min
        obj_height = y_max - y_min
    else:
        obj_width = w
        obj_height = h
        x_min, x_max, y_min, y_max = 0, w, 0, h

    if obj_width > obj_height:
        orientation_msg.append("Horizontal image detected, rotated 90° clockwise")
        output_arr = cv2.rotate(output_arr, cv2.ROTATE_90_CLOCKWISE)
        clean_mask = cv2.rotate(clean_mask, cv2.ROTATE_90_CLOCKWISE)

    # Re-calculate dimensions after potential rotation
    ys, xs = np.where(clean_mask > 100)
    if len(ys) > 0 and len(xs) > 0:
        y_min, y_max = int(ys.min()), int(ys.max())
        h_obj = y_max - y_min
        
        top_band = clean_mask[y_min: y_min + int(h_obj * 0.15)]
        bottom_band = clean_mask[y_max - int(h_obj * 0.15): y_max]
        
        top_px = np.where(top_band > 100)[1]
        bottom_px = np.where(bottom_band > 100)[1]
        
        top_width = int(top_px.max() - top_px.min()) if len(top_px) > 0 else 0
        bottom_width = int(bottom_px.max() - bottom_px.min()) if len(bottom_px) > 0 else 0
    else:
        top_width = 0
        bottom_width = 0

    if bottom_width < top_width:
        orientation_msg.append("Upside-down image detected, rotated 180°")
        output_arr = cv2.rotate(output_arr, cv2.ROTATE_180)
    else:
        orientation_msg.append("Orientation OK")

    output_image = Image.fromarray(output_arr, mode=mode)
    return output_image, orientation_msg, obj_width, obj_height, top_width, bottom_width


# ─────────────────────────────────────────────────────────────────────────────
# MODAL FUNCTION — GPU inference
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    gpu="T4",                        # Change to "A10G" or "A100" for faster inference
    timeout=600,
    volumes={CACHE_DIR: model_cache},
    secrets=[hf_secret] if hf_secret else [],
)
def segment_pants(
    image_bytes: bytes,
    overlay_color: tuple = (255, 105, 180),   # Pink
    opacity: float       = 0.35,
    transparent: bool    = True,
    skip_detection: bool = False,
    detection_text: str  = "pants. jeans. cargo pants. trousers.",
    conf_threshold: float = 0.35,
) -> dict:
    """
    Run the full pants segmentation pipeline on Modal GPU.
    """
    import torch
    from PIL import Image

    print(f"[Modal] GPU available: {torch.cuda.is_available()}")
    print(f"[Modal] Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Load input image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    print(f"[Modal] Input image size: {image.size}")

    num_detections = None

    # ── Step 1: Pants Detection ──────────────────────────────────────────────
    if not skip_detection:
        print("[Modal] Loading Grounding DINO…")
        try:
            dino_processor, dino_model, device = _load_dino()
        except Exception as e:
            return {"success": False, "message": f"Failed to load DINO: {e}"}

        print("[Modal] Running pants detection…")
        boxes, scores = _detect_pants(
            image, dino_processor, dino_model, device, detection_text, conf_threshold
        )

        if len(boxes) == 0:
            return {
                "success": False,
                "message": (
                    "No pants detected. Lower confidence threshold or "
                    "set skip_detection=True to bypass."
                ),
                "detections": 0,
            }

        num_detections = int(len(boxes))
        print(f"[Modal] Detected {num_detections} region(s), top score: {scores[0].item():.2f}")
        del dino_model, dino_processor  # free VRAM before loading RMBG

    # ── Step 2: Load RMBG-2.0 ───────────────────────────────────────────────
    print("[Modal] Loading RMBG-2.0…")
    try:
        rmbg_model, device = _load_rmbg()
    except Exception as e:
        return {"success": False, "message": f"Failed to load RMBG-2.0: {e}"}

    # ── Step 3: Remove Background ────────────────────────────────────────────
    print("[Modal] Running background removal…")
    try:
        original, clean_mask, _ = _run_rmbg(image, rmbg_model, device)
    except Exception as e:
        return {"success": False, "message": f"RMBG inference failed: {e}"}

    # ── Step 4: Apply Overlay & Fix Orientation ──────────────────────────────
    print("[Modal] Applying overlay & fixing orientation…")
    output_image, orient_msgs, obj_w, obj_h, top_w, bottom_w = _apply_overlay(
        original, clean_mask, overlay_color, opacity, transparent
    )

    # Encode output to PNG bytes
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    output_bytes = buf.getvalue()

    print(f"[Modal] Done! Output size: {len(output_bytes) / 1024:.1f} KB")

    # Commit model cache volume so weights persist across cold starts
    model_cache.commit()

    return {
        "success": True,
        "output_bytes": output_bytes,
        "message": "Segmentation complete.",
        "detections": num_detections,
        "orientation_msgs": orient_msgs,
        "obj_width": obj_w,
        "obj_height": obj_h,
        "top_width": top_w,
        "bottom_width": bottom_w,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI ASGI APP
# Exposes the frontend HTML and the post endpoint to process images.
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse

web_app = FastAPI(title="Slyte RMBG Segmentation Studio")


@web_app.get("/", response_class=HTMLResponse)
def read_root():
    import os
    # Serve index.html from container path, fallback to local path
    html_path = "/root/index.html"
    if not os.path.exists(html_path):
        html_path = "index.html"
    
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@web_app.post("/segment")
async def api_segment(
    file: UploadFile = File(...),
    overlay_color: str = Form("255,105,180"),
    opacity: float = Form(0.35),
    transparent: bool = Form(True),
    skip_detection: bool = Form(False),
    detection_text: str = Form("pants. jeans. cargo pants. trousers."),
    conf_threshold: float = Form(0.35),
):
    try:
        # Read the image bytes uploaded by the user
        image_bytes = await file.read()
        
        # Parse overlay_color (format: "r,g,b")
        try:
            rgb = tuple(map(int, overlay_color.split(",")))
            if len(rgb) != 3:
                raise ValueError()
        except Exception:
            rgb = (255, 105, 180) # Default to pink if parsing fails
            
        # Call segment_pants.local to run it inside the same container
        result = segment_pants.local(
            image_bytes=image_bytes,
            overlay_color=rgb,
            opacity=opacity,
            transparent=transparent,
            skip_detection=skip_detection,
            detection_text=detection_text,
            conf_threshold=conf_threshold,
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
            
        # Convert output bytes to base64 for inline JSON response
        encoded = base64.b64encode(result["output_bytes"]).decode("utf-8")
        
        return {
            "success": True,
            "image": f"data:image/png;base64,{encoded}",
            "message": result["message"],
            "detections": result["detections"],
            "orientation_msgs": result["orientation_msgs"],
            "obj_width": result["obj_width"],
            "obj_height": result["obj_height"],
            "top_width": result["top_width"],
            "bottom_width": result["bottom_width"],
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.function(
    gpu="T4",
    timeout=600,
    volumes={CACHE_DIR: model_cache},
    secrets=[hf_secret] if hf_secret else [],
)
@modal.asgi_app()
def fastapi_app():
    return web_app


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL ENTRYPOINT  —  modal run modal_test.py --image-path ./your_image.jpg
# ─────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    image_path: str  = "",
    output_path: str = "output.png",
    skip_detection: bool = False,
    transparent: bool    = True,
    opacity: float       = 0.35,
):
    """
    CLI entry point.
    """
    if not image_path:
        print("❌  Please provide --image-path <path-to-image>")
        return

    p = Path(image_path)
    if not p.exists():
        print(f"❌  File not found: {image_path}")
        return

    print(f"📤  Sending '{p.name}' to Modal GPU…")
    image_bytes = p.read_bytes()

    result = segment_pants.remote(
        image_bytes      = image_bytes,
        overlay_color    = (255, 105, 180),
        opacity          = opacity,
        transparent      = transparent,
        skip_detection   = skip_detection,
    )

    if result["success"]:
        out = Path(output_path)
        out.write_bytes(result["output_bytes"])
        det = result.get("detections")
        det_str = f", {det} region(s) detected" if det is not None else ""
        print(f"✅  Done{det_str}! Saved → {out.resolve()}")
    else:
        print(f"❌  Failed: {result['message']}")
