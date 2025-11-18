from __future__ import annotations

import io
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Luminara AI Upscaler Studio",
    page_icon=":sparkles:",
    layout="wide",
)

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    RESAMPLE_LANCZOS = Image.LANCZOS

BASE_MODELS_DIR = Path("models")
ENHANCEMENT_CHOICES = ("Edge sharpen", "Contrast boost", "Noise cleanup")
MIME_MAP = {
    "PNG": "image/png",
    "JPG": "image/jpeg",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
}
GLOBAL_STYLES = """
<style>
:root {
    --hero-gradient: radial-gradient(circle at 20% 20%, #5c6cfd 0%, #5b18d3 35%, #05060a 100%);
    --card-surface: rgba(255, 255, 255, 0.05);
    --border-glow: rgba(94, 207, 255, 0.35);
    --text-subtle: #9fb6ff;
    --accent: #6fffee;
    --warning: #ff9a62;
    --success: #92fba3;
}
.stApp {
    background: radial-gradient(circle at top, rgba(8, 10, 24, 0.92), #030308 65%, #010103 100%);
    color: #f6f7ff;
}
.hero-shell {
    padding: 2.2rem;
    border-radius: 28px;
    background: var(--hero-gradient);
    box-shadow: 0 35px 80px rgba(7, 5, 24, 0.65);
    position: relative;
    overflow: hidden;
    margin-bottom: 1.5rem;
}
.hero-eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.4em;
    font-size: 0.7rem;
    color: var(--accent);
    margin-bottom: 0.5rem;
}
.hero-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.4rem;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: var(--text-subtle);
    max-width: 540px;
}
.hero-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 0.9rem;
    margin-top: 1.5rem;
}
.stat-chip {
    position: relative;
    padding: 0.6rem 0 0.8rem;
}
.stat-chip::after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-glow), transparent);
}
.stat-chip h4 {
    margin: 0;
    font-size: 1.1rem;
    color: #fff;
}
.stat-chip p {
    margin: 0.2rem 0 0;
    color: var(--text-subtle);
    font-size: 0.9rem;
}
.hero-tagline {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.4rem 0.9rem;
    background: rgba(0, 0, 0, 0.25);
    border-radius: 999px;
    color: var(--success);
    border: 1px solid rgba(146, 251, 163, 0.3);
    margin-bottom: 1rem;
}
.mission-stream {
    display: flex;
    flex-wrap: wrap;
    gap: 1.8rem;
    margin-bottom: 1.5rem;
}
.mission-item {
    flex: 1;
    min-width: 240px;
    padding-left: 1.2rem;
    border-left: 2px solid rgba(146, 251, 163, 0.4);
}
.mission-item h3 {
    margin: 0 0 0.4rem;
    color: #fff;
}
.mission-item p,
.mission-item ol {
    color: var(--text-subtle);
    margin: 0;
    padding-left: 1rem;
}
.mission-item ol {
    list-style: decimal-leading-zero;
}
div[data-testid="stSidebar"] {
    background: rgba(2, 3, 10, 0.85);
    backdrop-filter: blur(18px);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
div[data-testid="stSidebar"] * {
    color: #d8e1ff !important;
}
div[data-testid="stSidebar"] .stButton>button {
    background: linear-gradient(120deg, #693bff, #00f3ff);
    border: none;
}
.timeline {
    border-left: 2px solid rgba(255, 255, 255, 0.08);
    margin-top: 1rem;
    padding-left: 1.5rem;
}
.timeline-entry {
    margin-bottom: 1.1rem;
    position: relative;
}
.timeline-entry::before {
    content: "";
    width: 12px;
    height: 12px;
    border-radius: 50%;
    border: 2px solid var(--accent);
    background: #030308;
    position: absolute;
    left: -1.75rem;
    top: 0.15rem;
}
.timeline-entry h4 {
    margin: 0;
    color: #fff;
}
.timeline-entry p {
    margin: 0.2rem 0 0;
    color: var(--text-subtle);
}
</style>
"""

MODEL_LIBRARY: Dict[str, Dict[str, Any]] = {
    "edsr": {
        "label": "EDSR",
        "model_key": "edsr",
        "description": (
            "High fidelity model that excels at photographic detail and textured surfaces."
        ),
        "quality_score": 5,
        "speed_score": 2,
        "tags": ("detail-focused", "photography"),
    },
    "espcn": {
        "label": "ESPCN",
        "model_key": "espcn",
        "description": (
            "Lightning-fast option ideal for previews, UI assets, or simple graphics."
        ),
        "quality_score": 3,
        "speed_score": 5,
        "tags": ("very fast", "preview"),
    },
    "fsrcnn": {
        "label": "FSRCNN",
        "model_key": "fsrcnn",
        "description": (
            "Balanced performer that keeps edges clean without sacrificing much speed."
        ),
        "quality_score": 4,
        "speed_score": 4,
        "tags": ("balanced", "general-purpose"),
    },
    "fsrcnn-small": {
        "label": "FSRCNN Small",
        "model_key": "fsrcnn",
        "description": (
            "Compact variant tuned for instant feedback with a small trade-off in detail."
        ),
        "quality_score": 3,
        "speed_score": 5,
        "tags": ("realtime", "lightweight"),
    },
    "lapsrn": {
        "label": "LapSRN",
        "model_key": "lapsrn",
        "description": (
            "Multiscale Laplacian network that preserves line art and crisp edges well."
        ),
        "quality_score": 4,
        "speed_score": 3,
        "tags": ("line-art", "sharp edges"),
    },
    "rcan": {
        "label": "RCAN",
        "model_key": "rcan",
        "description": (
            "Channel-attention network for premium texture preservation. Heavier runtime."
        ),
        "quality_score": 5,
        "speed_score": 1,
        "tags": ("textured", "premium"),
    },
}


@dataclass(frozen=True)
class ModelSpec:
    identifier: str
    display_name: str
    architecture: str
    model_key: str
    scale: int
    file_path: Path
    description: str
    quality_score: int
    speed_score: int
    tags: Tuple[str, ...]


def _extract_scale(label: str) -> int:
    digits = "".join(ch for ch in label if ch.isdigit())
    return int(digits) if digits else 1


def _infer_architecture(stem: str) -> str:
    name = stem.lower()
    if "fsrcnn-small" in name:
        return "fsrcnn-small"
    if "fsrcnn" in name:
        return "fsrcnn"
    if "lapsrn" in name:
        return "lapsrn"
    if "espcn" in name:
        return "espcn"
    if "edsr" in name:
        return "edsr"
    if "rcan" in name:
        return "rcan"
    return stem.lower()


def discover_model_specs(base_dir: Path) -> Dict[str, List[ModelSpec]]:
    registry: Dict[str, List[ModelSpec]] = {}
    if not base_dir.exists():
        return registry

    for scale_dir in sorted(base_dir.iterdir(), key=lambda p: _extract_scale(p.name)):
        if not scale_dir.is_dir():
            continue
        scale_label = scale_dir.name
        scale_value = _extract_scale(scale_label)
        specs: List[ModelSpec] = []

        for model_path in sorted(scale_dir.glob("*.pb")):
            architecture = _infer_architecture(model_path.stem)
            metadata = MODEL_LIBRARY.get(architecture, {})
            label = metadata.get("label", model_path.stem)
            spec = ModelSpec(
                identifier=f"{architecture}-{scale_label}",
                display_name=f"{label} ({scale_label})",
                architecture=architecture,
                model_key=str(metadata.get("model_key", architecture)),
                scale=scale_value,
                file_path=model_path,
                description=str(
                    metadata.get("description", "No description available yet.")
                ),
                quality_score=int(metadata.get("quality_score", 3)),
                speed_score=int(metadata.get("speed_score", 3)),
                tags=tuple(metadata.get("tags", ())),
            )
            specs.append(spec)

        if specs:
            specs.sort(key=lambda s: (-s.quality_score, s.speed_score, s.display_name))
            registry[scale_label] = specs

    return registry


def render_rating(score: int, maximum: int = 5) -> str:
    score = max(0, min(score, maximum))
    return f"[{'*' * score}{'.' * (maximum - score)}]"


def speed_label(score: int) -> str:
    lookup = {
        1: "Deliberate",
        2: "Steady",
        3: "Balanced",
        4: "Fast",
        5: "Lightning",
    }
    return lookup.get(score, "Balanced")


def format_resolution(shape: Sequence[int]) -> str:
    height, width = int(shape[0]), int(shape[1])
    megapixels = (width * height) / 1_000_000
    return f"{width} x {height} px ({megapixels:.2f} MP)"


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes = int(seconds // 60)
    remainder = seconds % 60
    return f"{minutes} min {remainder:.0f} s"


def format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def sanitize_filename(name: str) -> str:
    safe = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in name.strip()
    )
    safe = safe.strip("-_")
    return safe or "luminara"


@st.cache_data(show_spinner=False)
def get_model_registry(base_dir: str) -> Dict[str, List[ModelSpec]]:
    return discover_model_specs(Path(base_dir))


@st.cache_resource(show_spinner=False)
def load_sr_model(model_path: str, model_key: str, scale: int) -> cv2.dnn_superres.DnnSuperResImpl:
    model = cv2.dnn_superres.DnnSuperResImpl_create()
    model.readModel(model_path)
    model.setModel(model_key, scale)
    return model


def run_super_resolution(image_rgb: np.ndarray, spec: ModelSpec) -> np.ndarray:
    sr_model = load_sr_model(str(spec.file_path), spec.model_key, spec.scale)
    return sr_model.upsample(image_rgb)


def apply_post_processing(
    image_rgb: np.ndarray,
    enhancements: Sequence[str],
    sharpen_amount: float,
    contrast_amount: float,
    denoise_strength: int,
) -> np.ndarray:
    result = image_rgb.copy()

    if "Edge sharpen" in enhancements and sharpen_amount > 0:
        blur = cv2.GaussianBlur(result, ksize=(0, 0), sigmaX=3)
        result = cv2.addWeighted(result, 1 + sharpen_amount, blur, -sharpen_amount, 0)
        result = np.clip(result, 0, 255).astype(np.uint8)

    if "Contrast boost" in enhancements and contrast_amount > 0:
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clip_limit = float(np.interp(contrast_amount, [0.0, 1.0], [1.0, 3.5]))
        tile_value = int(np.interp(contrast_amount, [0.0, 1.0], [6, 8]))
        clahe = cv2.createCLAHE(
            clipLimit=max(1.0, clip_limit), tileGridSize=(max(2, tile_value),) * 2
        )
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge((l_channel, a_channel, b_channel))
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        result = np.clip(result, 0, 255).astype(np.uint8)

    if "Noise cleanup" in enhancements and denoise_strength > 0:
        strength = int(np.interp(denoise_strength, [1, 20], [5, 12]))
        result = cv2.fastNlMeansDenoisingColored(result, None, strength, strength, 7, 21)

    return np.clip(result, 0, 255).astype(np.uint8)


def encode_image(image_rgb: np.ndarray, fmt: str, quality: int) -> io.BytesIO:
    image = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    fmt_upper = fmt.upper()

    if fmt_upper in ("JPG", "JPEG"):
        image.save(
            buffer,
            format="JPEG",
            quality=max(10, min(100, quality)),
            subsampling=0,
            optimize=True,
        )
    elif fmt_upper == "PNG":
        image.save(buffer, format="PNG", compress_level=9)
    elif fmt_upper == "WEBP":
        image.save(
            buffer,
            format="WEBP",
            quality=max(10, min(100, quality)),
            method=6,
        )
    else:
        image.save(buffer, format=fmt_upper)

    buffer.seek(0)
    return buffer


def build_split_view(
    original_rgb: np.ndarray, upscaled_rgb: np.ndarray, split_ratio: float
) -> np.ndarray:
    target_h, target_w = upscaled_rgb.shape[:2]
    original_resized = cv2.resize(
        original_rgb, (target_w, target_h), interpolation=cv2.INTER_CUBIC
    )
    ratio = float(np.clip(split_ratio, 0.0, 1.0))
    split_col = max(1, min(target_w - 1, int(ratio * target_w)))
    canvas = upscaled_rgb.copy()
    canvas[:, :split_col, :] = original_resized[:, :split_col, :]

    line_start = max(split_col - 1, 0)
    line_end = min(split_col + 1, target_w)
    canvas[:, line_start:line_end, :] = (255, 128, 0)
    return canvas


def create_thumbnail(image_rgb: np.ndarray, max_edge: int = 320) -> np.ndarray:
    thumb = Image.fromarray(image_rgb)
    thumb.thumbnail((max_edge, max_edge), RESAMPLE_LANCZOS)
    return np.array(thumb)


def clamp(value: float, min_value: float = 0.0, max_value: float = 100.0) -> float:
    return float(max(min_value, min(max_value, value)))


def analyze_image_profile(image_rgb: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
    if image_rgb is None:
        return None

    image_float = image_rgb.astype(np.float32) / 255.0
    gray = cv2.cvtColor(image_float, cv2.COLOR_RGB2GRAY)

    brightness = clamp(float(gray.mean()) * 100)
    contrast = clamp(float(gray.std()) * 400)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    detail = clamp(float(laplacian.var()) * 10)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    noise = clamp(float(np.mean(np.abs(gray - blur))) * 800)

    r, g, b = cv2.split(image_float)
    rg = r - g
    yb = 0.5 * (r + g) - b
    rg_std, rg_mean = float(np.std(rg)), float(np.mean(rg))
    yb_std, yb_mean = float(np.std(yb)), float(np.mean(yb))
    color_metric = np.sqrt(rg_std**2 + yb_std**2) + 0.3 * np.sqrt(
        rg_mean**2 + yb_mean**2
    )
    color = clamp(color_metric * 40)

    return {
        "brightness": brightness,
        "contrast": contrast,
        "detail": detail,
        "noise": noise,
        "color": color,
    }


def build_image_radar(insights: Dict[str, float]) -> go.Figure:
    axes = ["Brightness", "Contrast", "Detail", "Vibrance", "Cleanliness"]
    values = [
        insights.get("brightness", 0.0),
        insights.get("contrast", 0.0),
        insights.get("detail", 0.0),
        insights.get("color", 0.0),
        clamp(100 - insights.get("noise", 0.0)),
    ]
    fig = go.Figure(
        data=go.Scatterpolar(
            r=values,
            theta=axes,
            fill="toself",
            name="Profile",
            line=dict(color="#7f9cff"),
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100], showticklabels=False)),
        showlegend=False,
        margin=dict(l=30, r=30, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_model_rows(model_registry: Dict[str, List[ModelSpec]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scale in sorted(model_registry.keys(), key=_extract_scale):
        for spec in model_registry[scale]:
            rows.append(
                {
                    "Scale": scale,
                    "Model": spec.display_name,
                    "Quality": spec.quality_score,
                    "Speed": spec.speed_score,
                    "Tags": ", ".join(spec.tags) if spec.tags else "None",
                    "Description": spec.description,
                }
            )
    return rows


def build_spec_personality(spec: ModelSpec) -> go.Figure:
    metrics = {
        "Fidelity": clamp((spec.quality_score / 5) * 100),
        "Velocity": clamp((spec.speed_score / 5) * 100),
        "Edge focus": clamp(((spec.quality_score + (5 - spec.speed_score)) / 10) * 100),
        "Versatility": clamp(((spec.quality_score + spec.speed_score) / 10) * 100),
    }
    fig = go.Figure(
        data=go.Scatterpolar(
            r=list(metrics.values()),
            theta=list(metrics.keys()),
            fill="toself",
            line=dict(color="#ffb347"),
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100], showticklabels=False)),
        showlegend=False,
        margin=dict(l=30, r=30, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def generate_intelligence_recommendations(
    insights: Optional[Dict[str, float]],
    active_spec: ModelSpec,
    selected_enhancements: Sequence[str],
    premium_spec: ModelSpec,
) -> List[str]:
    if not insights:
        return ["Load an image or use the demo asset to unlock AI guidance."]

    recs: List[str] = []

    if insights["noise"] > 55 and "Noise cleanup" not in selected_enhancements:
        recs.append("Image noise is elevated. Enable Noise cleanup for crisper results.")

    if insights["detail"] < 40 and active_spec.quality_score < premium_spec.quality_score:
        recs.append(
            f"Detail is limited. Consider switching to {premium_spec.display_name} "
            "for maximum fidelity."
        )

    if insights["contrast"] < 35 and "Contrast boost" not in selected_enhancements:
        recs.append("Contrast is subdued. Add Contrast boost for extra punch.")

    if insights["brightness"] < 30:
        recs.append("Scene is dark. Try a lower noise value plus Contrast boost.")

    if insights["color"] < 35:
        recs.append("Colors are muted. Export as PNG to keep every tone intact.")

    if not recs:
        recs.append("Profile looks balanced. Fire the upscale and review telemetry.")

    return recs
MODEL_REGISTRY = get_model_registry(str(BASE_MODELS_DIR))

if not MODEL_REGISTRY:
    st.error(
        "No trained models found. Place .pb files under models/<scale>/ before running."
    )
    st.stop()

available_scales = sorted(MODEL_REGISTRY.keys(), key=_extract_scale)
total_models = sum(len(specs) for specs in MODEL_REGISTRY.values())
max_scale_label = max(available_scales, key=_extract_scale)
fastest_spec = max(
    (spec for specs in MODEL_REGISTRY.values() for spec in specs),
    key=lambda spec: spec.speed_score,
)
quality_champion = max(
    (spec for specs in MODEL_REGISTRY.values() for spec in specs),
    key=lambda spec: spec.quality_score,
)
architecture_labels = sorted(
    {
        MODEL_LIBRARY.get(spec.architecture, {}).get(
            "label", spec.architecture.upper()
        )
        for specs in MODEL_REGISTRY.values()
        for spec in specs
    }
)

if "history" not in st.session_state:
    st.session_state["history"] = []
if "total_renders" not in st.session_state:
    st.session_state["total_renders"] = 0
if "pixels_processed" not in st.session_state:
    st.session_state["pixels_processed"] = 0
if "last_run" not in st.session_state:
    st.session_state["last_run"] = {}

st.markdown(GLOBAL_STYLES, unsafe_allow_html=True)


with st.sidebar:
    st.header("Mission controls")
    st.caption("Dial in the scale, architecture, and finishing lab before liftoff.")

    default_scale_index = (
        available_scales.index("4x") if "4x" in available_scales else 0
    )
    selected_scale = st.selectbox(
        "Scale factor",
        available_scales,
        index=default_scale_index,
    )

    scale_specs = MODEL_REGISTRY.get(selected_scale, [])
    if not scale_specs:
        st.error(f"No models available for {selected_scale}.")
        st.stop()

    recommended_spec = scale_specs[0]
    model_options = ["Auto (best quality)"] + [spec.display_name for spec in scale_specs]
    model_choice = st.selectbox(
        "Engine",
        model_options,
        help="Auto picks the sharpest model seen for this scale.",
    )

    if model_choice == "Auto (best quality)":
        active_spec = recommended_spec
    else:
        active_spec = next(spec for spec in scale_specs if spec.display_name == model_choice)

    st.markdown(f"**{active_spec.display_name}** ready for {active_spec.scale}x lift.")
    st.caption(active_spec.description)
    st.caption(
        f"Quality {render_rating(active_spec.quality_score)} | "
        f"Speed {speed_label(active_spec.speed_score)}"
    )
    if active_spec.tags:
        st.caption("Tags: " + ", ".join(active_spec.tags))

    with st.expander("Enhancement lab", expanded=True):
        selected_enhancements = list(
            st.multiselect(
                "Enhancements",
                ENHANCEMENT_CHOICES,
                default=("Edge sharpen",),
                help="Stack extra passes tuned for upscaled imagery.",
            )
        )
        sharpen_amount = 0.0
        contrast_amount = 0.0
        denoise_strength = 0

        if "Edge sharpen" in selected_enhancements:
            sharpen_amount = st.slider(
                "Sharpen strength",
                min_value=0.05,
                max_value=1.0,
                value=0.25,
                step=0.05,
            )
        if "Contrast boost" in selected_enhancements:
            contrast_amount = st.slider(
                "Contrast lift",
                min_value=0.05,
                max_value=1.0,
                value=0.2,
                step=0.05,
            )
        if "Noise cleanup" in selected_enhancements:
            denoise_strength = st.slider(
                "Denoise level",
                min_value=1,
                max_value=20,
                value=6,
                help="Higher values remove more noise but may soften details.",
            )

    with st.expander("Output lab", expanded=True):
        output_format = st.selectbox(
            "Download format",
            ("PNG", "JPG", "WEBP"),
            index=0,
        ).upper()
        if output_format in ("JPG", "WEBP"):
            output_quality = st.slider(
                "Quality",
                min_value=70,
                max_value=100,
                value=95,
                step=5,
            )
        else:
            output_quality = 100
        custom_filename = st.text_input(
            "File name",
            value="luminara-upscaled",
            help="Base name without extension.",
        )

    demo_path = Path("result.jpeg")
    if demo_path.exists():
        if st.button("Load demo image"):
            st.session_state["demo_image_bytes"] = demo_path.read_bytes()
            st.session_state["demo_image_name"] = demo_path.name

    with st.expander("Flight recorder", expanded=False):
        history: List[Dict[str, Any]] = st.session_state.get("history", [])
        if not history:
            st.caption("Processed images will drop here for quick re-downloads.")
        else:
            for idx, entry in enumerate(history):
                output_h, output_w = entry["output_resolution"]
                st.markdown(
                    f"**{entry['filename']}**  \n"
                    f"{entry['model']} | {output_w} x {output_h}px | {format_duration(entry['duration'])}"
                )
                if entry.get("enhancements"):
                    st.caption("Enhancements: " + ", ".join(entry["enhancements"]))
                if entry.get("thumbnail") is not None:
                    st.image(entry["thumbnail"], use_column_width=True)
                st.download_button(
                    "Download again",
                    data=entry["bytes"],
                    file_name=entry["filename"],
                    mime=MIME_MAP.get(entry["format"], "image/png"),
                    key=f"history-download-{idx}",
                )
                st.markdown("---")
            if st.button("Clear history"):
                st.session_state["history"] = []


session_launches = st.session_state["total_renders"]
pixels_processed = st.session_state["pixels_processed"]
pixels_label = (
    f"{pixels_processed / 1_000_000:.1f} MP" if pixels_processed else "0 MP"
)
architectures_copy = ", ".join(architecture_labels)

hero_html = f"""
<div class="hero-shell">
    <div class="hero-tagline">&gt;&gt; Live engine: {active_spec.display_name} - {active_spec.scale}x</div>
    <div class="hero-eyebrow">Luminara Mission Control</div>
    <div class="hero-title">Ultra upscale studio</div>
    <p class="hero-subtitle">
        Fully local command center featuring {len(architecture_labels)} neural architectures.
        Models auto-curate to keep fidelity high while staying fast. Selected scale: <strong>{selected_scale}</strong>.
    </p>
    <div class="hero-grid">
        <div class="stat-chip">
            <p>Models online</p>
            <h4>{total_models}</h4>
        </div>
        <div class="stat-chip">
            <p>Scales ready</p>
            <h4>{len(available_scales)} ({max_scale_label})</h4>
        </div>
        <div class="stat-chip">
            <p>Fast lane</p>
            <h4>{fastest_spec.display_name}</h4>
        </div>
        <div class="stat-chip">
            <p>Flagship quality</p>
            <h4>{quality_champion.display_name}</h4>
        </div>
        <div class="stat-chip">
            <p>Session launches</p>
            <h4>{session_launches}</h4>
        </div>
        <div class="stat-chip">
            <p>Pixels uplifted</p>
            <h4>{pixels_label}</h4>
        </div>
    </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

enhancement_snapshot = ", ".join(selected_enhancements) if selected_enhancements else "None"
mission_html = f"""
<div class="mission-stream">
    <div class="mission-item">
        <h3>Flight plan</h3>
        <ol>
            <li>Deploy a source image or quick-load the demo asset.</li>
            <li>Dial in the scale + engine from the mission controls.</li>
            <li>Stack enhancements, ignite the upscale, then audit results.</li>
        </ol>
    </div>
    <div class="mission-item">
        <h3>Current brief</h3>
        <p><strong>Architectures online:</strong> {architectures_copy}</p>
        <p><strong>Enhancements armed:</strong> {enhancement_snapshot}</p>
        <p><strong>Output target:</strong> {output_format} @ {output_quality}</p>
    </div>
</div>
"""
st.markdown(mission_html, unsafe_allow_html=True)


image_bytes: Optional[bytes] = None
image_label = ""
image_size_bytes = 0
image_bgr: Optional[np.ndarray] = None
image_rgb: Optional[np.ndarray] = None

studio_tab, intelligence_tab, launch_pad_tab = st.tabs(
    ["Studio console", "Image intelligence", "Model launchpad"]
)

with studio_tab:
    st.subheader("Studio console")
    st.markdown(
        "Upload imagery, preview the upscale footprint, and fire the render rocket."
    )

    uploaded_file = st.file_uploader(
        "Upload an image (PNG, JPG, JPEG, BMP, WEBP, TIFF)",
        type=["png", "jpg", "jpeg", "bmp", "webp", "tiff"],
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        image_label = uploaded_file.name or "uploaded-image"
        image_size_bytes = getattr(uploaded_file, "size", len(image_bytes))
    elif st.session_state.get("demo_image_bytes"):
        image_bytes = st.session_state["demo_image_bytes"]
        image_label = st.session_state.get("demo_image_name", "demo-image")
        image_size_bytes = len(image_bytes)

    if image_bytes:
        as_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(as_array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            st.error("Could not decode the image. Please try a different file.")
            image_bytes = None
        else:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if image_rgb is not None and image_bgr is not None:
        input_h, input_w = image_rgb.shape[:2]
        target_w = input_w * active_spec.scale
        target_h = input_h * active_spec.scale
        target_megapixels = (target_w * target_h) / 1_000_000

        st.image(
            image_rgb,
            caption=f"{image_label} - {input_w} x {input_h}px",
            use_column_width=True,
        )

        stats_cols = st.columns(4)
        stats_cols[0].metric("Input width", f"{input_w}px")
        stats_cols[1].metric("Input height", f"{input_h}px")
        stats_cols[2].metric("Input size", format_size(image_size_bytes))
        stats_cols[3].metric("Estimated output", f"{target_w} x {target_h}px")

        if image_size_bytes > 15 * 1024 * 1024:
            st.warning(
                "The input file is quite large. Upscaling might take longer than usual."
            )
        if target_megapixels > 40:
            st.warning(
                f"The requested output is about {target_megapixels:.1f} MP. "
                "Consider a lower scale if processing becomes slow."
            )

        st.markdown(
            f"Recommended model: **{recommended_spec.display_name}** "
            f"(Quality {render_rating(recommended_spec.quality_score)}, "
            f"Speed {speed_label(recommended_spec.speed_score)})."
        )

        with st.expander(f"Model lineup for {selected_scale}", expanded=False):
            for spec in scale_specs:
                st.markdown(
                    f"- **{spec.display_name}** - Quality {render_rating(spec.quality_score)}, "
                    f"Speed {speed_label(spec.speed_score)}  \n  {spec.description}"
                )

        run_button = st.button("Ignite upscale", type="primary")
    else:
        run_button = st.button("Ignite upscale", type="primary", disabled=True)

    if run_button and image_rgb is not None and image_bgr is not None:
        timeline_events: List[Dict[str, Any]] = []
        with st.spinner("Upscaling in progress..."):
            sr_start = time.perf_counter()
            upscaled_rgb = run_super_resolution(image_rgb, active_spec)
            sr_duration = time.perf_counter() - sr_start
            timeline_events.append(
                {
                    "label": "Super resolution",
                    "duration": sr_duration,
                    "detail": f"{active_spec.display_name} @ {active_spec.scale}x",
                }
            )

            if selected_enhancements:
                enh_start = time.perf_counter()
                upscaled_rgb = apply_post_processing(
                    upscaled_rgb,
                    selected_enhancements,
                    sharpen_amount,
                    contrast_amount,
                    denoise_strength,
                )
                enh_duration = time.perf_counter() - enh_start
                timeline_events.append(
                    {
                        "label": "Enhancements",
                        "duration": enh_duration,
                        "detail": ", ".join(selected_enhancements),
                    }
                )

            encode_start = time.perf_counter()
            buffer = encode_image(upscaled_rgb, output_format, output_quality)
            encode_duration = time.perf_counter() - encode_start
            timeline_events.append(
                {
                    "label": "Encoding",
                    "duration": encode_duration,
                    "detail": output_format,
                }
            )

        elapsed = sum(event["duration"] for event in timeline_events)
        throughput = (
            (upscaled_rgb.shape[0] * upscaled_rgb.shape[1])
            / max(elapsed, 1e-6)
            / 1_000_000
        )

        st.success(
            f"Upscaled with {active_spec.display_name} in {format_duration(elapsed)}."
        )

        st.session_state["total_renders"] += 1
        st.session_state["pixels_processed"] += (
            upscaled_rgb.shape[0] * upscaled_rgb.shape[1]
        )
        st.session_state["last_run"] = {
            "timeline": timeline_events,
            "elapsed": elapsed,
            "throughput": throughput,
            "model": active_spec.display_name,
            "scale": active_spec.scale,
            "enhancements": list(selected_enhancements),
        }

        output_bytes = buffer.getvalue()
        extension = "jpg" if output_format in ("JPG", "JPEG") else output_format.lower()

        base_name_source = custom_filename or Path(image_label).stem
        safe_base_name = sanitize_filename(base_name_source)
        download_name = f"{safe_base_name}_{active_spec.scale}x.{extension}"
        mime_type = MIME_MAP.get(output_format, "application/octet-stream")

        result_tab, comparison_tab, details_tab, telemetry_tab = st.tabs(
            ["Upscaled image", "Comparison", "Details", "Telemetry"]
        )

        with result_tab:
            st.image(
                upscaled_rgb,
                caption=f"Upscaled result ({download_name})",
                use_column_width=True,
            )
            st.download_button(
                "Download upscaled image",
                data=output_bytes,
                file_name=download_name,
                mime=mime_type,
                type="primary",
            )

        with comparison_tab:
            split_ratio = st.slider(
                "Reveal original vs. upscaled",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05,
            )
            comparison_image = build_split_view(image_rgb, upscaled_rgb, split_ratio)
            st.image(
                comparison_image,
                caption="Left shows the original, right shows the upscaled result.",
                use_column_width=True,
            )

        with details_tab:
            st.json(
                {
                    "Model": active_spec.display_name,
                    "Enhancements": selected_enhancements or "None",
                    "Input resolution": format_resolution(image_rgb.shape[:2]),
                    "Output resolution": format_resolution(upscaled_rgb.shape[:2]),
                    "Scale": f"{active_spec.scale}x",
                    "Processing time": format_duration(elapsed),
                    "Output size": format_size(len(output_bytes)),
                    "Output format": output_format,
                }
            )

        with telemetry_tab:
            timeline_html = "<div class='timeline'>"
            for event in st.session_state.get("last_run", {}).get("timeline", []):
                timeline_html += (
                    "<div class='timeline-entry'>"
                    f"<h4>{event['label']}</h4>"
                    f"<p>{event['detail']} | {format_duration(event['duration'])}</p>"
                    "</div>"
                )
            timeline_html += "</div>"
            st.markdown(timeline_html, unsafe_allow_html=True)

            st.metric(
                "Throughput",
                f"{st.session_state['last_run'].get('throughput', 0):.2f} MP/s",
                delta=None,
            )
            st.caption(
                f"Total runtime {format_duration(st.session_state['last_run'].get('elapsed', 0.0))}."
            )

        history_entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": active_spec.display_name,
            "enhancements": list(selected_enhancements),
            "input_resolution": image_rgb.shape[:2],
            "output_resolution": upscaled_rgb.shape[:2],
            "duration": elapsed,
            "bytes": output_bytes,
            "format": output_format,
            "filename": download_name,
            "thumbnail": create_thumbnail(upscaled_rgb),
        }

        history = st.session_state.get("history", [])
        history.insert(0, history_entry)
        st.session_state["history"] = history[:5]

with intelligence_tab:
    st.subheader("Image intelligence")
    if image_rgb is None:
        st.info("Upload an image or load the demo asset to unlock insights.")
    else:
        insights = analyze_image_profile(image_rgb)
        if not insights:
            st.warning("Insights could not be generated for this image.")
        else:
            metrics_cols = st.columns(4)
            metrics_cols[0].metric("Brightness", f"{insights['brightness']:.0f}/100")
            metrics_cols[1].metric("Contrast", f"{insights['contrast']:.0f}/100")
            metrics_cols[2].metric("Detail", f"{insights['detail']:.0f}/100")
            metrics_cols[3].metric("Noise level", f"{insights['noise']:.0f}/100")

            radar_fig = build_image_radar(insights)
            st.plotly_chart(radar_fig, use_container_width=True)

            st.markdown("**Mission guidance**")
            for rec in generate_intelligence_recommendations(
                insights, active_spec, selected_enhancements, quality_champion
            ):
                st.markdown(f"- {rec}")

            st.caption(
                "Noise scores closer to 0 indicate cleaner content. Cleanliness in the radar uses 100 - noise."
            )

with launch_pad_tab:
    st.subheader("Model launchpad")
    model_rows = build_model_rows(MODEL_REGISTRY)
    if not model_rows:
        st.info("Model analytics will appear here once weights are discovered.")
    else:
        model_df = pd.DataFrame(model_rows)
        st.dataframe(model_df, use_container_width=True)

        st.markdown("**Active engine personality**")
        persona_fig = build_spec_personality(active_spec)
        st.plotly_chart(persona_fig, use_container_width=True)

        st.caption(
            f"{active_spec.display_name} balances quality {render_rating(active_spec.quality_score)} "
            f"with speed {speed_label(active_spec.speed_score)}."
        )
