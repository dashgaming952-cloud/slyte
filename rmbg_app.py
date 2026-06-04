# -*- coding: utf-8 -*-
"""
RMBG-2.0 Streamlit App
Pants Detection → Background Removal → Pink Overlay → Auto-Orientation Fix
Converted from Google Colab notebook.
"""

import os
import io
import time
import threading

# ── Set HF token at startup ───────────────────────────────────────────────────
if "HF_TOKEN" not in os.environ:
    try:
        import streamlit as st
        if "HF_TOKEN" in st.secrets:
            os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
    except Exception:
        pass

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from transformers import (
    AutoModelForImageSegmentation,
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RMBG-2.0 · Pants Segmentation",
    page_icon="👖",
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
        background: linear-gradient(135deg, #0f0c29 0%, #1a1040 50%, #24243e 100%);
        min-height: 100vh;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #ff69b4 0%, #c44dff 50%, #7b2fff 100%);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(255,105,180,0.25);
    }
    .hero-banner h1 {
        color: white;
        font-size: 2.6rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        color: rgba(255,255,255,0.85);
        font-size: 1.05rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    /* Step cards */
    .step-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
        transition: border-color 0.3s;
    }
    .step-card:hover {
        border-color: rgba(255,105,180,0.4);
    }
    .step-title {
        color: #ff69b4;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    .step-desc {
        color: rgba(255,255,255,0.75);
        font-size: 0.9rem;
    }

    /* Result badges */
    .badge-success {
        display: inline-block;
        background: rgba(72,199,116,0.15);
        color: #48c774;
        border: 1px solid rgba(72,199,116,0.3);
        border-radius: 30px;
        padding: 0.3rem 1rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .badge-warn {
        display: inline-block;
        background: rgba(255,180,0,0.15);
        color: #ffb400;
        border: 1px solid rgba(255,180,0,0.3);
        border-radius: 30px;
        padding: 0.3rem 1rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .badge-error {
        display: inline-block;
        background: rgba(255,69,58,0.15);
        color: #ff453a;
        border: 1px solid rgba(255,69,58,0.3);
        border-radius: 30px;
        padding: 0.3rem 1rem;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* Image captions */
    .img-caption {
        text-align: center;
        color: rgba(255,255,255,0.5);
        font-size: 0.8rem;
        margin-top: 0.4rem;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #ff69b4, #c44dff) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.6rem 2rem !important;
        font-size: 1rem !important;
        transition: all 0.3s !important;
        box-shadow: 0 4px 20px rgba(255,105,180,0.35) !important;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(255,105,180,0.5) !important;
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #ff69b4, #c44dff) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.6rem 2rem !important;
        font-size: 1rem !important;
        width: 100%;
        transition: all 0.3s !important;
        box-shadow: 0 4px 20px rgba(255,105,180,0.3) !important;
    }

    /* Section label */
    .section-label {
        color: rgba(255,255,255,0.4);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 0.5rem;
    }

    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.08) !important;
    }

    /* Info box override */
    .stAlert {
        border-radius: 12px !important;
    }

    /* Download progress box */
    .dl-box {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,105,180,0.25);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0 1rem 0;
    }
    .dl-title {
        color: #ff69b4;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
    }
    .dl-bytes {
        color: rgba(255,255,255,0.85);
        font-size: 1.1rem;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
    }
    .dl-sub {
        color: rgba(255,255,255,0.45);
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
        <h1>👖 RMBG-2.0 · Pants Segmentation</h1>
        <p>Detect pants → Remove background → Apply pink overlay → Auto-fix orientation</p>
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
        help="Required to download gated models like RMBG-2.0",
    )
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    st.markdown("---")
    st.markdown('<div class="section-label">🎨 Overlay Color</div>', unsafe_allow_html=True)
    overlay_color_name = st.selectbox(
        label="Overlay Color",
        options=["Pink 🩷", "Blue 💙", "Green 💚", "Orange 🧡", "Custom"],
        label_visibility="collapsed",
    )
    color_map = {
        "Pink 🩷":   (255, 105, 180),
        "Blue 💙":   (0,   180, 255),
        "Green 💚":  (50,  205, 50),
        "Orange 🧡": (255, 140, 0),
    }
    if overlay_color_name == "Custom":
        hex_color = st.color_picker("Pick a color", "#FF69B4")
        h = hex_color.lstrip("#")
        overlay_color = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    else:
        overlay_color = color_map[overlay_color_name]

    st.markdown("---")
    st.markdown('<div class="section-label">🌫️ Overlay Opacity</div>', unsafe_allow_html=True)
    opacity = st.slider("Opacity", 0.0, 1.0, 0.35, 0.05, label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="section-label">🎯 Detection Confidence</div>', unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence", 0.1, 0.9, 0.35, 0.05, label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="section-label">🔍 Detection Labels</div>', unsafe_allow_html=True)
    detection_text = st.text_input(
        "Labels (dot-separated)",
        value="pants. jeans. cargo pants. trousers.",
        label_visibility="collapsed",
        help='e.g. "pants. jeans. trousers."',
    )

    st.markdown("---")
    skip_pants_check = st.checkbox(
        "⏭️ Skip pants detection (run RMBG on any image)",
        value=False,
    )

    st.markdown("---")
    st.markdown('<div class="section-label">💾 Output Format</div>', unsafe_allow_html=True)
    output_format = st.radio(
        "Format",
        ["PNG (transparent)", "PNG (white background)"],
        label_visibility="collapsed",
    )

# ─────────────────────────────────────────────────────────────────────────────
# HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("ℹ️ How it works", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            '<div class="step-card"><div class="step-title">Step 1</div>'
            '<div class="step-desc">🔍 Grounding DINO detects pants in the image</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="step-card"><div class="step-title">Step 2</div>'
            '<div class="step-desc">✂️ RMBG-2.0 generates a precise segmentation mask</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="step-card"><div class="step-title">Step 3</div>'
            '<div class="step-desc">🎨 Color overlay is blended onto the masked region</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            '<div class="step-card"><div class="step-title">Step 4</div>'
            '<div class="step-desc">🔄 Orientation is auto-corrected using waist-width heuristic</div></div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# THREAD-SAFE PROGRESS STORE
# ─────────────────────────────────────────────────────────────────────────────
# Background thread writes here; main thread polls and renders UI.

_progress_lock = threading.Lock()
_progress_data = {
    "active": False,
    "model_name": "",
    "desc": "",
    "n": 0,          # bytes downloaded
    "total": 0,      # total bytes (0 = unknown)
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


# ─────────────────────────────────────────────────────────────────────────────
# TQDM PATCH  (runs in background thread – only writes to plain dict, no st.*)
# ─────────────────────────────────────────────────────────────────────────────

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
        # Write to thread-safe dict (no Streamlit calls here!)
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
    # huggingface_hub uses tqdm.auto — patch that too
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
    # Restore tqdm.auto as well
    try:
        import tqdm.auto as _tqdm_auto
        _tqdm_auto.tqdm.__init__ = _original_tqdm_init
        _tqdm_auto.tqdm.update   = _original_tqdm_update
        _tqdm_auto.tqdm.close    = _original_tqdm_close
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND LOADING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _cached_load_rmbg():
    model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-2.0",
        trust_remote_code=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, device


@st.cache_resource(show_spinner=False)
def _cached_load_dino():
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino = dino.to(device)
    return processor, dino, device


def _load_in_thread(load_fn, model_name):
    """
    Run *load_fn* in a daemon thread while this function polls and updates
    the Streamlit UI from the **main thread**.

    Returns the result of load_fn(), or raises on error.
    """
    _reset_progress()
    with _progress_lock:
        _progress_data["model_name"] = model_name

    _install_patch()

    result_holder = {}

    def _worker():
        try:
            result_holder["value"] = load_fn()
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

    # ── UI containers (created on main thread) ──
    header_slot  = st.empty()
    bar_slot     = st.empty()
    text_slot    = st.empty()

    header_slot.markdown(
        f'<div class="dl-box"><div class="dl-title">📥 Downloading {model_name}…</div>'
        f'<div class="dl-bytes">Starting…</div>'
        f'<div class="dl-sub">Please wait, this only happens once</div></div>',
        unsafe_allow_html=True,
    )

    # ── Polling loop on the main thread ──
    while t.is_alive():
        time.sleep(0.25)  # poll every 250 ms

        with _progress_lock:
            n_bytes = _progress_data["n"]
            total_bytes = _progress_data["total"]
            desc = _progress_data["desc"] or "Downloading"

        n_mb    = n_bytes    / (1024 * 1024)
        tot_mb  = total_bytes / (1024 * 1024)

        if total_bytes > 0:
            pct   = min(n_bytes / total_bytes, 1.0)
            label = f"{n_mb:.1f} MB / {tot_mb:.0f} MB  ({pct*100:.0f}%)"
            bar_slot.progress(pct)
        elif n_bytes > 0:
            label = f"{n_mb:.1f} MB downloaded"
            # indeterminate spinner style
            bar_slot.progress(0.0)
        else:
            label = "Connecting…"
            bar_slot.progress(0.0)

        header_slot.markdown(
            f'<div class="dl-box">'
            f'<div class="dl-title">📥 Downloading {model_name}…</div>'
            f'<div class="dl-bytes">{label}</div>'
            f'<div class="dl-sub">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Thread finished – clean up UI
    header_slot.empty()
    bar_slot.empty()
    text_slot.empty()

    with _progress_lock:
        err = _progress_data.get("error")
        res = _progress_data.get("result")

    if err is not None:
        raise err
    return res


def load_dino():
    return _load_in_thread(_cached_load_dino, "Grounding DINO (~800 MB)")


def load_rmbg():
    return _load_in_thread(_cached_load_rmbg, "RMBG-2.0 (~200 MB)")


# ─────────────────────────────────────────────────────────────────────────────
# CORE PROCESSING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def detect_pants(image: Image.Image, processor, dino, device, text: str, threshold: float):
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs.input_ids,
        target_sizes=[image.size[::-1]],
    )
    boxes = results[0]["boxes"]
    scores = results[0]["scores"]
    valid = scores > threshold
    boxes = boxes[valid]
    scores = scores[valid]
    return boxes, scores


def run_rmbg(image: Image.Image, rmbg_model, device):
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])
    original = np.array(image)
    original_size = image.size
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = rmbg_model(input_tensor)[-1]
    mask = pred[0].squeeze().cpu().numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask = cv2.resize(mask, original_size)
    mask = (mask * 255).astype(np.uint8)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contour found in the segmentation mask.")
    largest = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, cv2.FILLED)
    clean_mask = cv2.GaussianBlur(clean_mask, (7, 7), 0)
    return original, clean_mask, original_size


def apply_overlay_and_orientation(
    original: np.ndarray,
    clean_mask: np.ndarray,
    overlay_color: tuple,
    opacity: float,
    transparent: bool,
):
    h, w = clean_mask.shape
    original = cv2.resize(original, (w, h))
    overlay = np.zeros_like(original)
    overlay[:] = overlay_color

    alpha = clean_mask.astype(np.float32) / 255.0
    alpha_3d = np.expand_dims(alpha, axis=2)

    if transparent:
        result = (
            original.astype(np.float32) * (1 - opacity * alpha_3d)
            + overlay.astype(np.float32) * (opacity * alpha_3d)
        ).astype(np.uint8)
        rgba = np.dstack([result, clean_mask])
        edges = cv2.Canny(clean_mask, 100, 200)
        rgba[edges > 0] = [overlay_color[0], overlay_color[1], overlay_color[2], 255]
        output_arr = rgba
        mode = "RGBA"
    else:
        white_bg = np.ones((h, w, 3), dtype=np.float32) * 255
        foreground = original.astype(np.float32) * alpha_3d + white_bg * (1 - alpha_3d)
        result = np.where(
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
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    obj_width = x_max - x_min
    obj_height = y_max - y_min

    if obj_width > obj_height:
        orientation_msg.append("↩️ Horizontal → rotated 90° clockwise")
        if mode == "RGBA":
            output_arr = cv2.rotate(output_arr, cv2.ROTATE_90_CLOCKWISE)
        else:
            output_arr = cv2.rotate(output_arr, cv2.ROTATE_90_CLOCKWISE)
        clean_mask = cv2.rotate(clean_mask, cv2.ROTATE_90_CLOCKWISE)

    ys, xs = np.where(clean_mask > 100)
    y_min, y_max = int(ys.min()), int(ys.max())
    h_obj = y_max - y_min
    top_band = clean_mask[y_min: y_min + int(h_obj * 0.15)]
    bottom_band = clean_mask[y_max - int(h_obj * 0.15): y_max]
    top_px = np.where(top_band > 100)[1]
    bottom_px = np.where(bottom_band > 100)[1]
    top_width = int(top_px.max() - top_px.min()) if len(top_px) > 0 else 0
    bottom_width = int(bottom_px.max() - bottom_px.min()) if len(bottom_px) > 0 else 0

    if bottom_width < top_width:
        orientation_msg.append("🔃 Upside-down → rotated 180°")
        output_arr = cv2.rotate(output_arr, cv2.ROTATE_180)
    else:
        orientation_msg.append("✅ Orientation OK")

    output_image = Image.fromarray(output_arr, mode=mode)
    return output_image, orientation_msg, obj_width, obj_height, top_width, bottom_width


def image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI – IMAGE UPLOAD
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

image = Image.open(uploaded).convert("RGB")
orig_w, orig_h = image.size

col_prev, col_info = st.columns([2, 1])
with col_prev:
    st.image(image, caption="Input Image", width='stretch')
with col_info:
    st.markdown(f"**Filename:** `{uploaded.name}`")
    st.markdown(f"**Dimensions:** `{orig_w} × {orig_h}` px")
    st.markdown(f"**Size:** `{uploaded.size / 1024:.1f} KB`")
    st.markdown(f"**Device:** `{'CUDA (GPU) 🚀' if torch.cuda.is_available() else 'CPU 🖥️'}`")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# RUN BUTTON
# ─────────────────────────────────────────────────────────────────────────────

run_btn = st.button("🚀 Run Segmentation", type="primary", width='stretch')

if not run_btn:
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

if not hf_token and not os.environ.get("HF_TOKEN"):
    st.error(
        "⚠️ **HF Token missing!** Please enter your Hugging Face token in the sidebar. "
        "Get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)."
    )
    st.stop()

status = st.empty()
progress = st.progress(0)
log_box = st.empty()
logs = []


def log(msg):
    logs.append(msg)
    log_box.markdown("\n\n".join(f"• {l}" for l in logs))


# ── Step 1: Pants Detection ───────────────────────────────────────────────────

if not skip_pants_check:
    status.markdown("**Step 1/4 — Loading Grounding DINO…**")
    progress.progress(5)

    try:
        dino_processor, dino_model, device = load_dino()
    except Exception as e:
        st.error(f"❌ Failed to load Grounding DINO: {e}")
        st.stop()

    log("✅ Grounding DINO loaded")
    status.markdown("**Step 1/4 — Detecting pants…**")
    progress.progress(20)

    with st.spinner("Detecting pants…"):
        try:
            boxes, scores = detect_pants(
                image, dino_processor, dino_model, device, detection_text, conf_threshold
            )
        except Exception as e:
            st.error(f"❌ Detection error: {e}")
            st.stop()

    if len(boxes) == 0:
        st.error(
            "❌ **No pants detected** in this image. "
            "Try lowering the confidence threshold or enabling 'Skip pants detection' in the sidebar."
        )
        progress.progress(0)
        st.stop()

    log(f"✅ Pants detected — {len(boxes)} region(s), top score: `{scores[0].item():.2f}`")
    progress.progress(30)
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("⏭️ Pants detection skipped")
    progress.progress(30)

# ── Step 2: Load RMBG ────────────────────────────────────────────────────────

status.markdown("**Step 2/4 — Loading RMBG-2.0…**")
progress.progress(35)

try:
    rmbg_model, device = load_rmbg()
except Exception as e:
    st.error(f"❌ Failed to load RMBG-2.0: {e}")
    st.stop()

log("✅ RMBG-2.0 loaded")
progress.progress(50)

# ── Step 3: Segmentation ─────────────────────────────────────────────────────

status.markdown("**Step 3/4 — Running background removal…**")

with st.spinner("Running RMBG-2.0 inference…"):
    try:
        original_np, clean_mask, original_size = run_rmbg(image, rmbg_model, device)
    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()

log("✅ Segmentation mask generated")
progress.progress(70)

# ── Step 4: Overlay + Orientation ────────────────────────────────────────────

status.markdown("**Step 4/4 — Applying overlay & fixing orientation…**")
transparent = output_format == "PNG (transparent)"

with st.spinner("Compositing result…"):
    output_img, orient_msgs, obj_w, obj_h, top_w, bottom_w = apply_overlay_and_orientation(
        original_np, clean_mask, overlay_color, opacity, transparent
    )

for m in orient_msgs:
    log(m)

progress.progress(100)
status.empty()

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### 🎉 Results")

res_col1, res_col2 = st.columns(2)
with res_col1:
    st.image(image, caption="Original", width='stretch')
    st.markdown('<div class="img-caption">Input image</div>', unsafe_allow_html=True)

with res_col2:
    if transparent:
        bg = Image.new("RGB", output_img.size, (255, 255, 255))
        bg.paste(output_img, mask=output_img.split()[3])
        display_img = bg
    else:
        display_img = output_img
    st.image(display_img, caption="Segmented Output", width='stretch')
    st.markdown('<div class="img-caption">RMBG-2.0 output</div>', unsafe_allow_html=True)

# ── Stats ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### 📊 Diagnostics")

d1, d2, d3, d4 = st.columns(4)
with d1:
    st.metric("Object Width", f"{obj_w} px")
with d2:
    st.metric("Object Height", f"{obj_h} px")
with d3:
    st.metric("Top Width", f"{top_w} px")
with d4:
    st.metric("Bottom Width", f"{bottom_w} px")

orient_label = "Upside-down corrected" if "180°" in "\n".join(orient_msgs) else (
    "Rotated 90°" if "90°" in "\n".join(orient_msgs) else "No rotation needed"
)
orient_status = "badge-warn" if "rotated" in orient_label.lower() else "badge-success"
st.markdown(
    f'<span class="{orient_status}">{orient_label}</span>',
    unsafe_allow_html=True,
)

# ── Download ─────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### 💾 Download")

out_bytes = image_to_bytes(output_img)
base_name = os.path.splitext(uploaded.name)[0]
dl_filename = f"{base_name}_rmbg_output.png"

st.download_button(
    label="⬇️ Download Output PNG",
    data=out_bytes,
    file_name=dl_filename,
    mime="image/png",
    width='stretch',
)

st.markdown(
    "<br><div style='text-align:center; color:rgba(255,255,255,0.3); font-size:0.8rem;'>"
    "Powered by <b>RMBG-2.0</b> (briaai) · <b>Grounding DINO</b> (IDEA-Research) · <b>Streamlit</b>"
    "</div>",
    unsafe_allow_html=True,
)
