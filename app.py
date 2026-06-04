# -*- coding: utf-8 -*-
"""
RMBG-2.0 Streamlit App - CPU Optimized for Render
"""

import os
import io
import time
import gc
import threading
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# ── Set HF token at startup ───────────────────────────────────────────────────
if "HF_TOKEN" not in os.environ:
    try:
        if "HF_TOKEN" in st.secrets:
            os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RMBG-2.0 · Background Removal",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0813 0%, #110b29 50%, #17123d 100%);
        min-height: 100vh;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.03);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #c44dff 0%, #7b2fff 50%, #ff69b4 100%);
        border-radius: 16px;
        padding: 2rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(123, 47, 255, 0.2);
    }
    .hero-banner h1 {
        color: white;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        color: rgba(255, 255, 255, 0.85);
        font-size: 1rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    /* Info Badges */
    .badge-cpu {
        display: inline-block;
        background: rgba(0, 180, 255, 0.15);
        color: #00b4ff;
        border: 1px solid rgba(0, 180, 255, 0.3);
        border-radius: 30px;
        padding: 0.3rem 1rem;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .badge-success {
        display: inline-block;
        background: rgba(72, 199, 116, 0.15);
        color: #48c774;
        border: 1px solid rgba(72, 199, 116, 0.3);
        border-radius: 30px;
        padding: 0.3rem 1rem;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* Image captions */
    .img-caption {
        text-align: center;
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.8rem;
        margin-top: 0.4rem;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #7b2fff, #c44dff) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.6rem 2rem !important;
        font-size: 1rem !important;
        transition: all 0.3s !important;
        box-shadow: 0 4px 20px rgba(123, 47, 255, 0.3) !important;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(123, 47, 255, 0.45) !important;
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7b2fff, #ff69b4) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.6rem 2rem !important;
        font-size: 1rem !important;
        width: 100%;
        transition: all 0.3s !important;
        box-shadow: 0 4px 20px rgba(123, 47, 255, 0.25) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 25px rgba(123, 47, 255, 0.4) !important;
    }

    /* Section label */
    .section-label {
        color: rgba(255, 255, 255, 0.4);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 0.5rem;
    }

    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
    }

    /* Download progress box */
    .dl-box {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(123, 47, 255, 0.25);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0 1rem 0;
    }
    .dl-title {
        color: #c44dff;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
    }
    .dl-bytes {
        color: rgba(255, 255, 255, 0.85);
        font-size: 1.1rem;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
    }
    .dl-sub {
        color: rgba(255, 255, 255, 0.45);
        font-size: 0.8rem;
        margin-top: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <h1>✨ RMBG-2.0 · Background Removal</h1>
        <p>Extract subjects instantly with CPU-optimized state-of-the-art AI</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR – SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    st.markdown('<div class="section-label">🔑 Hugging Face Token</div>', unsafe_allow_html=True)
    hf_token = st.text_input(
        label="HF Token",
        value=os.environ.get("HF_TOKEN", ""),
        type="password",
        placeholder="hf_xxxxxxxxxxxx",
        label_visibility="collapsed",
        help="Required to download the gated RMBG-2.0 model weights. Can be left blank if HF_TOKEN is configured in Render environment variables.",
    )
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    st.markdown("---")
    st.markdown('<div class="section-label">📏 Max Process Resolution</div>', unsafe_allow_html=True)
    max_resolution = st.selectbox(
        label="Max Resolution",
        options=["768px (Fast & Low RAM 🖥️)", "1024px (High Quality)"],
        index=0,
        label_visibility="collapsed",
        help="Resizes large images before processing to ensure fast inference and prevent memory issues on Render.",
    )
    max_res_val = 768 if "768" in max_resolution else 1024

    st.markdown("---")
    st.markdown('<div class="section-label">🎯 Extraction Options</div>', unsafe_allow_html=True)
    isolate_largest = st.checkbox(
        "Isolate single largest object",
        value=False,
        help="If checked, only the single largest foreground object is kept. Useful for product or single-item cataloging.",
    )

    st.markdown("---")
    st.markdown('<div class="section-label">💾 Background Mode</div>', unsafe_allow_html=True)
    bg_mode = st.radio(
        "Background Mode",
        ["Transparent PNG", "White Background", "Black Background"],
        label_visibility="collapsed",
    )

# ─────────────────────────────────────────────────────────────────────────────
# THREAD-SAFE PROGRESS STORE FOR HF DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
_progress_lock = threading.Lock()
_progress_data = {
    "active": False,
    "model_name": "",
    "desc": "",
    "n": 0,          # bytes downloaded
    "total": 0,      # total bytes
    "done": False,
    "error": None,
    "result": None,
}

def _reset_progress():
    with _progress_lock:
        _progress_data.update({
            "active": True,
            "model_name": "",
            "desc": "",
            "n": 0,
            "total": 0,
            "done": False,
            "error": None,
            "result": None,
        })

def _finish_progress(result=None, error=None):
    with _progress_lock:
        _progress_data["active"] = False
        _progress_data["done"] = True
        _progress_data["result"] = result
        _progress_data["error"] = error

# ── TQDM Patching for Hugging Face Hub Downloads ─────────────────────────────
import tqdm as _tqdm_module

_original_tqdm_init   = _tqdm_module.tqdm.__init__
_original_tqdm_update = _tqdm_module.tqdm.update
_original_tqdm_close  = _tqdm_module.tqdm.close
_active_bars: dict = {}

def _patched_init(self, *args, **kwargs):
    _original_tqdm_init(self, *args, **kwargs)
    desc = kwargs.get("desc", "") or getattr(self, "desc", "") or ""
    _active_bars[id(self)] = {"desc": desc, "n": self.n or 0, "total": self.total or 0}

def _patched_update(self, n=1):
    res = _original_tqdm_update(self, n)
    bar = _active_bars.get(id(self))
    if bar is not None:
        bar["n"] = self.n or 0
        bar["total"] = self.total or 0
        bar["desc"] = getattr(self, "desc", "") or bar["desc"]
        with _progress_lock:
            _progress_data["n"] = bar["n"]
            _progress_data["total"] = bar["total"]
            _progress_data["desc"] = bar["desc"]
    return res

def _patched_close(self):
    _active_bars.pop(id(self), None)
    return _original_tqdm_close(self)

def _install_patch():
    _tqdm_module.tqdm.__init__  = _patched_init
    _tqdm_module.tqdm.update    = _patched_update
    _tqdm_module.tqdm.close     = _patched_close
    try:
        import tqdm.auto as _tqdm_auto
        _tqdm_auto.tqdm.__init__ = _patched_init
        _tqdm_auto.tqdm.update   = _patched_update
        _tqdm_auto.tqdm.close    = _patched_close
    except Exception:
        pass

def _remove_patch():
    _tqdm_module.tqdm.__init__  = _original_tqdm_init
    _tqdm_module.tqdm.update    = _original_tqdm_update
    _tqdm_module.tqdm.close     = _original_tqdm_close
    try:
        import tqdm.auto as _tqdm_auto
        _tqdm_auto.tqdm.__init__ = _original_tqdm_init
        _tqdm_auto.tqdm.update   = _original_tqdm_update
        _tqdm_auto.tqdm.close    = _original_tqdm_close
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _cached_load_rmbg():
    # Force cpu device execution
    model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-2.0",
        trust_remote_code=True,
    )
    model = model.to("cpu")
    model.eval()
    return model

def load_rmbg():
    """
    Loads RMBG-2.0 model on a background daemon thread while showing
    download progress on the Streamlit main thread.
    """
    _reset_progress()
    with _progress_lock:
        _progress_data["model_name"] = "RMBG-2.0 (~176 MB)"

    _install_patch()
    result_holder = {}

    def _worker():
        try:
            result_holder["value"] = _cached_load_rmbg()
        except Exception as exc:
            result_holder["error"] = exc
        finally:
            _remove_patch()
            _finish_progress(
                result=result_holder.get("value"),
                error=result_holder.get("error"),
            )

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    header_slot = st.empty()
    bar_slot    = st.empty()

    header_slot.markdown(
        '<div class="dl-box"><div class="dl-title">📥 Downloading RMBG-2.0…</div>'
        '<div class="dl-bytes">Connecting…</div>'
        '<div class="dl-sub">Downloading weights from Hugging Face Hub (cached after first run)</div></div>',
        unsafe_allow_html=True,
    )

    while t.is_alive():
        time.sleep(0.25)
        with _progress_lock:
            n_bytes = _progress_data["n"]
            total_bytes = _progress_data["total"]
            desc = _progress_data["desc"] or "Downloading"

        n_mb = n_bytes / (1024 * 1024)
        tot_mb = total_bytes / (1024 * 1024)

        if total_bytes > 0:
            pct = min(n_bytes / total_bytes, 1.0)
            label = f"{n_mb:.1f} MB / {tot_mb:.1f} MB  ({pct*100:.0f}%)"
            bar_slot.progress(pct)
        elif n_bytes > 0:
            label = f"{n_mb:.1f} MB downloaded"
            bar_slot.progress(0.0)
        else:
            label = "Connecting to Hugging Face Hub…"
            bar_slot.progress(0.0)

        header_slot.markdown(
            f'<div class="dl-box">'
            f'<div class="dl-title">📥 Downloading RMBG-2.0…</div>'
            f'<div class="dl-bytes">{label}</div>'
            f'<div class="dl-sub">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    header_slot.empty()
    bar_slot.empty()

    with _progress_lock:
        err = _progress_data.get("error")
        res = _progress_data.get("result")

    if err is not None:
        raise err
    return res

# ─────────────────────────────────────────────────────────────────────────────
# CORE BACKGROUND REMOVAL PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────
def run_rmbg_cpu(image: Image.Image, rmbg_model, isolate_largest: bool) -> Image.Image:
    # 1024x1024 is the architecture dimension that RMBG-2.0 performs best on
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])
    
    # Keep track of working image size
    w, h = image.size
    
    # Run transform - explicitly uses CPU tensor allocation
    input_tensor = transform(image).unsqueeze(0).to("cpu")
    
    with torch.inference_mode():
        # inference step
        pred = rmbg_model(input_tensor)[-1]
    
    # Process outputs on CPU
    mask = pred[0].squeeze().cpu().numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    
    # Resize mask back to working image size
    mask = cv2.resize(mask, (w, h))
    mask = (mask * 255).astype(np.uint8)
    
    # Binarize mask
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    if isolate_largest:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            clean_mask = np.zeros_like(mask)
            cv2.drawContours(clean_mask, [largest], -1, 255, cv2.FILLED)
            mask = clean_mask
            
    # Morphological clean up and edge smoothing
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Build transparent output
    img_np = np.array(image)
    rgba = np.dstack([img_np, mask])
    
    # Immediate cleanup of temporary arrays & tensors
    del input_tensor, pred, mask, img_np
    gc.collect()
    
    return Image.fromarray(rgba, "RGBA")

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

def image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    # If image has an alpha channel, save as PNG, else JPEG to keep it small
    if img.mode == "RGBA":
        img.save(buf, format="PNG")
    else:
        img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UIFLOW
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📂 Upload Image")
uploaded = st.file_uploader(
    label="Drop an image here",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    label_visibility="collapsed",
)

if not uploaded:
    st.info("👆 Upload an image to get started.")
    st.stop()

# 1. Open original image
original_full = Image.open(uploaded).convert("RGB")
orig_w, orig_h = original_full.size

# 2. Immediately resize to optimized constraints (lowers memory pressure)
working_image = resize_image_aspect(original_full, max_res_val)
work_w, work_h = working_image.size

col_prev, col_info = st.columns([2, 1])
with col_prev:
    st.image(working_image, caption="Input Image Preview", use_column_width=True)
with col_info:
    st.markdown(f"**Filename:** `{uploaded.name}`")
    st.markdown(f"**Original Dimensions:** `{orig_w} × {orig_h}` px")
    st.markdown(f"**Processing Dimensions:** `{work_w} × {work_h}` px")
    st.markdown(f"**File Size:** `{uploaded.size / 1024:.1f} KB`")
    st.markdown(f"**Device Mode:** <span class='badge-cpu'>CPU ONLY 🖥️</span>", unsafe_allow_html=True)

st.markdown("---")

# ── RUN BUTTON ───────────────────────────────────────────────────────────────
run_btn = st.button("🚀 Process Background Removal", type="primary", use_container_width=True)

if run_btn:
    # ── Token validations before start ──────────────────────────────────────────
    if not os.environ.get("HF_TOKEN"):
        st.error(
            "⚠️ **Hugging Face Token is missing!** Bria AI's RMBG-2.0 is a gated repository. "
            "Please configure your Hugging Face Token in the sidebar settings or set it as a "
            "`HF_TOKEN` Environment Variable on Render."
        )
        st.stop()

    status = st.empty()
    progress = st.progress(0)

    # Step 1: Loading model
    status.markdown("**Step 1/2 — Loading RMBG-2.0 Model…**")
    progress.progress(15)

    try:
        rmbg_model = load_rmbg()
    except Exception as e:
        st.error(
            f"❌ **Failed to load RMBG-2.0:** {e}\n\n"
            "This error is usually caused by an invalid/missing Hugging Face Token or terms agreement. "
            "Ensure your HF token is valid and you have accepted the model license terms at "
            "[huggingface.co/briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)."
        )
        progress.progress(0)
        st.stop()

    # Step 2: Running Inference
    status.markdown("**Step 2/2 — Removing Background on CPU…**")
    progress.progress(60)

    try:
        with st.spinner("Processing image..."):
            rgba_output = run_rmbg_cpu(working_image, rmbg_model, isolate_largest)
        
        # Build composite based on settings
        if bg_mode == "White Background":
            bg = Image.new("RGBA", rgba_output.size, (255, 255, 255, 255))
            final_output = Image.alpha_composite(bg, rgba_output).convert("RGB")
        elif bg_mode == "Black Background":
            bg = Image.new("RGBA", rgba_output.size, (0, 0, 0, 255))
            final_output = Image.alpha_composite(bg, rgba_output).convert("RGB")
        else:
            final_output = rgba_output

        progress.progress(100)
        status.markdown("<span class='badge-success'>✅ Complete! See results below.</span>", unsafe_allow_html=True)
        time.sleep(1)
        status.empty()
        progress.empty()

        # Display Results
        st.markdown("---")
        st.markdown("### 🎉 Results")

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.image(working_image, caption="Original Image", use_column_width=True)
        with res_col2:
            st.image(final_output, caption="Result", use_column_width=True)

        # Download option
        st.markdown("---")
        st.markdown("### 💾 Download")

        out_format = "PNG" if bg_mode == "Transparent PNG" else "JPEG"
        out_bytes = image_to_bytes(final_output)
        base_name = os.path.splitext(uploaded.name)[0]
        dl_filename = f"{base_name}_no_bg.{out_format.lower()}"

        st.download_button(
            label=f"⬇️ Download {out_format} Result",
            data=out_bytes,
            file_name=dl_filename,
            mime=f"image/{out_format.lower()}",
            use_container_width=True,
        )

        # Cleanup results from session memory
        del final_output, rgba_output, out_bytes
        gc.collect()

    except Exception as e:
        st.error(f"❌ **An error occurred during processing:** {e}")
        progress.progress(0)
        status.empty()

st.markdown(
    "<br><div style='text-align:center; color:rgba(255,255,255,0.3); font-size:0.8rem;'>"
    "Powered by <b>RMBG-2.0</b> (briaai) · <b>Streamlit</b> · CPU-Optimized Server Deployment"
    "</div>",
    unsafe_allow_html=True,
)
