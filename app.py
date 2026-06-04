# -*- coding: utf-8 -*-
"""
RMBG-2.0 FastAPI Backend - CPU Optimized for Render
"""

import os
import io
import gc
import base64
import threading
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# ── FastAPI App Setup ────────────────────────────────────────────────────────
app = FastAPI(
    title="RMBG-2.0 Background Remover API",
    description="CPU-optimized background removal API utilizing Bria AI's RMBG-2.0 model.",
    version="1.0.0"
)

# Enable CORS so local static HTML files can query the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Set HF token from Streamlit secrets (for fallback compatibility) ─────────
if "HF_TOKEN" not in os.environ:
    try:
        import streamlit as st
        if "HF_TOKEN" in st.secrets:
            os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
    except Exception:
        pass

# ── Thread-Safe Global Model Caching ──────────────────────────────────────────
model_lock = threading.Lock()
rmbg_model = None
model_ready = False      # True once model is loaded and ready
model_error: Optional[str] = None  # Holds error string if pre-load failed

def get_model(hf_token_override: Optional[str] = None):
    """
    Lazy-loads the RMBG-2.0 model thread-safely.
    This prevents Render deployment timeout errors during the initial boot phase
    by downloading model weights on the first request rather than at server startup.
    """
    global rmbg_model
    with model_lock:
        if rmbg_model is None:
            # Determine which Hugging Face token to use
            token = hf_token_override or os.environ.get("HF_TOKEN")
            if not token:
                raise ValueError(
                    "Hugging Face Token is missing. Please configure 'HF_TOKEN' as an "
                    "environment variable in your Render dashboard, or supply it as the "
                    "'X-HF-Token' request header."
                )
            
            # Temporarily configure environment token for transformers library
            os.environ["HF_TOKEN"] = token
            
            # Load the gated RMBG-2.0 model on CPU
            rmbg_model = AutoModelForImageSegmentation.from_pretrained(
                "briaai/RMBG-2.0",
                trust_remote_code=True,
            )
            rmbg_model = rmbg_model.to("cpu")
            rmbg_model.eval()
            global model_ready
            model_ready = True

        return rmbg_model

# ── Image Processors ─────────────────────────────────────────────────────────
def resize_image_aspect(image: Image.Image, max_size: int) -> Image.Image:
    w, h = image.size
    if w <= max_size and h <= max_size:
        return image
    if w > h:
        new_w = max_size
        new_h = int(h * (max_size / w))
    else:
        new_h = max_size
        new_w = int(w * (max_size / h))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

def run_rmbg_cpu(image: Image.Image, rmbg_model, isolate_largest: bool) -> Image.Image:
    # 1024x1024 is the standard resolution RMBG-2.0 is trained on
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])
    
    w, h = image.size
    input_tensor = transform(image).unsqueeze(0).to("cpu")
    
    with torch.inference_mode():
        pred = rmbg_model(input_tensor)[-1]
    
    # Process prediction masks on CPU
    mask = pred[0].squeeze().cpu().numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    
    # Resize mask to current image dimensions
    mask = cv2.resize(mask, (w, h))
    mask = (mask * 255).astype(np.uint8)
    
    # Thresholding
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Isolate largest single contour if requested
    if isolate_largest:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            clean_mask = np.zeros_like(mask)
            cv2.drawContours(clean_mask, [largest], -1, 255, cv2.FILLED)
            mask = clean_mask
            
    # Smoothing edges
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Composite transparent RGBA image
    img_np = np.array(image)
    rgba = np.dstack([img_np, mask])
    
    # Explicit memory cleanup
    del input_tensor, pred, mask, img_np
    gc.collect()
    
    return Image.fromarray(rgba, "RGBA")

# ── Startup: Pre-load model in background thread ────────────────────────────
def _preload_model_background():
    """Runs model loading in a background thread so the app starts instantly."""
    global model_error
    try:
        print("[startup] Beginning background model pre-load...")
        get_model()
        print("[startup] Model pre-loaded and ready.")
    except Exception as exc:
        model_error = str(exc)
        print(f"[startup] Background model pre-load failed: {exc}")

@app.on_event("startup")
async def startup_event():
    thread = threading.Thread(target=_preload_model_background, daemon=True)
    thread.start()

# ── API Endpoints ────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    """
    Health check endpoint for Render monitoring.
    """
    return {
        "status": "online",
        "message": "RMBG-2.0 Background Remover API is healthy and running.",
        "model_ready": model_ready,
        "device": "cpu"
    }

@app.get("/status")
def model_status():
    """
    Lightweight polling endpoint for the frontend to check if the model
    has finished loading. Returns model_ready=true once the model is warm.
    """
    if model_error:
        return {"model_ready": False, "error": model_error}
    return {"model_ready": model_ready, "error": None}

@app.post("/remove-bg")
async def remove_bg(
    file: UploadFile = File(...),
    max_resolution: int = Form(768),
    isolate_largest: bool = Form(False),
    bg_mode: str = Form("Transparent PNG"),
    x_hf_token: Optional[str] = Header(None, alias="X-HF-Token")
):
    """
    Removes the background from the uploaded image.
    """
    try:
        # 1. Read uploaded image bytes
        file_bytes = await file.read()
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        
        # 2. Lazy-load model (checks headers first, falls back to env vars)
        try:
            model = get_model(x_hf_token)
        except ValueError as val_err:
            raise HTTPException(status_code=401, detail=str(val_err))
        except Exception as load_err:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to load RMBG-2.0 model weights: {str(load_err)}. "
                       "Please verify your Hugging Face Token permissions and repository acceptance."
            )
            
        # 3. Apply low-RAM aspect-ratio resizing before running inference
        working_img = resize_image_aspect(image, max_resolution)
        work_w, work_h = working_img.size
        
        # 4. Perform background removal on CPU
        rgba_output = run_rmbg_cpu(working_img, model, isolate_largest)
        
        # 5. Apply requested background styling
        if bg_mode == "White Background":
            bg = Image.new("RGBA", rgba_output.size, (255, 255, 255, 255))
            final_img = Image.alpha_composite(bg, rgba_output).convert("RGB")
        elif bg_mode == "Black Background":
            bg = Image.new("RGBA", rgba_output.size, (0, 0, 0, 255))
            final_img = Image.alpha_composite(bg, rgba_output).convert("RGB")
        else:
            final_img = rgba_output
            
        # 6. Encode image to Base64 data URL
        buf = io.BytesIO()
        if final_img.mode == "RGBA":
            final_img.save(buf, format="PNG")
            mime = "image/png"
        else:
            final_img.save(buf, format="JPEG", quality=90)
            mime = "image/jpeg"
            
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:{mime};base64,{img_base64}"
        
        # 7. Final garbage collection
        del file_bytes, image, working_img, rgba_output, final_img, buf
        gc.collect()
        
        return {
            "success": True,
            "image": data_url,
            "width": work_w,
            "height": work_h
        }
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error occurred: {str(exc)}")
