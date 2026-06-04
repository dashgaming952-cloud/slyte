# -*- coding: utf-8 -*-
"""
modal_test.py
RMBG-2.0 Pants Segmentation — Modal GPU Deployment

Run locally:
    modal run modal_test.py --image-path ./your_image.jpg

Deploy as persistent function:
    modal deploy modal_test.py
"""

import io
import os
from pathlib import Path

import modal

# ─────────────────────────────────────────────────────────────────────────────
# MODAL IMAGE
# All dependencies baked into the container image.
# ─────────────────────────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "transformers>=4.40.0",
        "Pillow",
        "opencv-python-headless",   # headless — no GUI needed in container
        "kornia",
        "huggingface_hub",
        "numpy",
        "accelerate",
        "timm",
    )
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
        return Image.fromarray(rgba, "RGBA")
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
        return Image.fromarray(result, "RGB")


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

    Args:
        image_bytes:     Raw bytes of the input image (JPEG / PNG / etc.)
        overlay_color:   RGB tuple for the overlay tint (default: pink)
        opacity:         Overlay opacity 0.0–1.0
        transparent:     True = RGBA PNG output; False = white-bg RGB PNG
        skip_detection:  Skip Grounding DINO pants check
        detection_text:  Dot-separated labels for DINO detection
        conf_threshold:  Minimum detection confidence

    Returns:
        dict with keys:
            "success"       bool
            "output_bytes"  PNG bytes of the result image (if success)
            "message"       Status / error message
            "detections"    Number of detected regions (if detection ran)
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

    # ── Step 4: Apply Overlay ────────────────────────────────────────────────
    print("[Modal] Applying overlay…")
    output_image = _apply_overlay(original, clean_mask, overlay_color, opacity, transparent)

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
    }


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

    Examples:
        modal run modal_test.py --image-path ./pants.jpg
        modal run modal_test.py --image-path ./pants.jpg --skip-detection
        modal run modal_test.py --image-path ./pants.jpg --output-path ./result.png
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
