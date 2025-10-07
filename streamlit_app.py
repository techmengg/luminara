from __future__ import annotations

import io
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
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

if "history" not in st.session_state:
    st.session_state["history"] = []


with st.sidebar:
    st.header("Controls")

    default_scale_index = (
        available_scales.index("4x") if "4x" in available_scales else 0
    )
    selected_scale = st.selectbox(
        "Upscale factor",
        available_scales,
        index=default_scale_index,
    )

    scale_specs = MODEL_REGISTRY.get(selected_scale, [])
    if not scale_specs:
        st.error(f"No models available for {selected_scale}.")
        st.stop()

    recommended_spec = scale_specs[0]
    model_options = ["Auto (best quality)"] + [spec.display_name for spec in scale_specs]
    model_choice = st.selectbox("AI model", model_options)

    if model_choice == "Auto (best quality)":
        active_spec = recommended_spec
    else:
        active_spec = next(spec for spec in scale_specs if spec.display_name == model_choice)

    st.caption(active_spec.description)
    st.text(f"Quality focus: {render_rating(active_spec.quality_score)}")
    st.text(f"Speed profile: {speed_label(active_spec.speed_score)}")
    if active_spec.tags:
        st.caption("Tags: " + ", ".join(active_spec.tags))

    with st.expander("Optional enhancements", expanded=True):
        selected_enhancements = list(
            st.multiselect(
                "Enhancements",
                ENHANCEMENT_CHOICES,
                default=("Edge sharpen",),
                help="Add gentle post-processing tuned for upscaled imagery.",
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

    with st.expander("Output settings", expanded=True):
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

    with st.expander("Recent renders", expanded=False):
        history: List[Dict[str, Any]] = st.session_state.get("history", [])
        if not history:
            st.caption("Your processed images will appear here.")
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


st.title("Luminara AI Upscaler Studio")
st.caption("Streamlined AI upscaling with model intelligence, guided tuning, and quick comparisons.")

summary_cols = st.columns(3)
summary_cols[0].metric("Models detected", str(total_models))
summary_cols[1].metric("Max upscale", max_scale_label)
summary_cols[2].metric("Fastest model", fastest_spec.display_name)

st.markdown(
    """
    1. Upload or pick a demo image.
    2. Choose a scale and model; keep Auto for the recommended pick.
    3. Apply optional enhancements, then run the upscale and download the result.
    """,
)


uploaded_file = st.file_uploader(
    "Upload an image (PNG, JPG, JPEG, BMP, WEBP, TIFF)",
    type=["png", "jpg", "jpeg", "bmp", "webp", "tiff"],
)

image_bytes: Optional[bytes] = None
image_label = ""
image_size_bytes = 0

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    image_label = uploaded_file.name or "uploaded-image"
    image_size_bytes = getattr(uploaded_file, "size", len(image_bytes))
elif st.session_state.get("demo_image_bytes"):
    image_bytes = st.session_state["demo_image_bytes"]
    image_label = st.session_state.get("demo_image_name", "demo-image")
    image_size_bytes = len(image_bytes)

image_bgr: Optional[np.ndarray] = None
image_rgb: Optional[np.ndarray] = None

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

    st.subheader("Preview")
    st.image(
        image_rgb,
        caption=f"{image_label} — {input_w} x {input_h}px",
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
                f"- **{spec.display_name}** — Quality {render_rating(spec.quality_score)}, "
                f"Speed {speed_label(spec.speed_score)}  \n  {spec.description}"
            )

    run_button = st.button("Upscale image", type="primary")
else:
    run_button = st.button("Upscale image", type="primary", disabled=True)


if run_button and image_rgb is not None and image_bgr is not None:
    with st.spinner("Upscaling in progress..."):
        start_time = time.perf_counter()
        upscaled_rgb = run_super_resolution(image_rgb, active_spec)
        if selected_enhancements:
            upscaled_rgb = apply_post_processing(
                upscaled_rgb,
                selected_enhancements,
                sharpen_amount,
                contrast_amount,
                denoise_strength,
            )
        elapsed = time.perf_counter() - start_time

    st.success(
        f"Upscaled with {active_spec.display_name} in {format_duration(elapsed)}."
    )

    buffer = encode_image(upscaled_rgb, output_format, output_quality)
    output_bytes = buffer.getvalue()
    extension = "jpg" if output_format in ("JPG", "JPEG") else output_format.lower()

    base_name_source = custom_filename or Path(image_label).stem
    safe_base_name = sanitize_filename(base_name_source)
    download_name = f"{safe_base_name}_{active_spec.scale}x.{extension}"
    mime_type = MIME_MAP.get(output_format, "application/octet-stream")

    result_tab, comparison_tab, details_tab = st.tabs(
        ["Upscaled image", "Comparison", "Details"]
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
