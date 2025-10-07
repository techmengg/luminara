# Luminara - AI Upscaler Studio

Luminara is an open-source, local AI image upscaler built with Python, Streamlit, and OpenCV's DNN Super Resolution. It auto-discovers pretrained models (EDSR/ESPCN/FSRCNN/LapSRN/RCAN) by scale, recommends the best option, and provides a clean interface with split-view comparison, guided enhancements (sharpen/contrast/denoise), multi-format downloads, and a render history for quick re-downloads.

Processing runs locally - no images leave your machine.

---

## Features

- Model auto-discovery from `models/<scale>/*.pb` with smart defaults
- Supports multiple architectures: EDSR, ESPCN, FSRCNN (and small), LapSRN, RCAN
- "Auto (best quality)" model selection per scale with quality/speed hints
- Optional enhancements tuned for upscaled imagery:
  - Edge sharpening with halo-resistant blend
  - Contrast lift via CLAHE for natural pop
  - Noise cleanup using fast non-local means
- Split-view comparison: drag a slider to reveal before/after
- Multi-format downloads: PNG, JPG, WEBP (in-memory; no temp files)
- Render history: thumbnails plus one-click re-download of recent results
- Helpful warnings for very large outputs and performance tips
- Clean, wide layout optimized for both laptops and desktops

---

## Quick Start

Requirements:

- Python 3.9+ (3.10 or 3.11 recommended)
- Platform: Windows, macOS, or Linux

Install and run:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

---

## Model Weights and Folder Layout

Luminara discovers models from the repository's `models` directory. Use one subfolder per scale (for example `2x`, `3x`, `4x`, `8x`) and place the `.pb` files inside. Example:

```
models/
  2x/
    EDSR_x2.pb
    ESPCN_x2.pb
    FSRCNN-small_x2.pb
    FSRCNN_x2.pb
    LapSRN_x2.pb
  3x/
    EDSR_x3.pb
    ESPCN_x3.pb
    FSRCNN-small_x3.pb
    FSRCNN_x3.pb
  4x/
    EDSR_x4.pb
    ESPCN_x4.pb
    FSRCNN-small_x4.pb
    FSRCNN_x4.pb
    LapSRN_x4.pb
  8x/
    LapSRN_x8.pb
```

Luminara infers architecture by filename (case-insensitive):

- Contains `edsr` -> EDSR
- Contains `espcn` -> ESPCN
- Contains `fsrcnn-small` -> FSRCNN Small
- Contains `fsrcnn` -> FSRCNN
- Contains `lapsrn` -> LapSRN
- Contains `rcan` -> RCAN

You can add new `.pb` weights at any time; they appear automatically the next time you load the app.

---

## Usage Guide

1. Upload an image (PNG, JPG, JPEG, BMP, WEBP, TIFF) or load the demo image if provided.
2. Pick an upscale factor (2x, 3x, 4x, or 8x). Leave "Auto (best quality)" enabled to use the recommended model for that scale, or choose a specific architecture.
3. Optional: enable enhancements and adjust their sliders.
4. Optional: choose output format (PNG, JPG, or WEBP) and file name.
5. Click "Upscale image".
6. Review results in tabs:
   - Upscaled image (with download button)
   - Comparison (split-view original vs upscaled)
   - Details (model, timing, resolutions, output file size)
7. Recent renders appear in the sidebar; quickly re-download or clear history.

Tips:

- Start with 4x for small images; try 2x or 3x for moderate photos.
- EDSR and RCAN generally yield the best photographic detail but run slower.
- FSRCNN and ESPCN are much faster and good for previews or simple graphics.
- LapSRN often keeps line art and edges crisp.

---

## Quality vs Speed Cheat-Sheet

| Model        | Quality (1-5) | Speed Profile | Good For                            |
|--------------|---------------|---------------|-------------------------------------|
| EDSR         | 5             | Deliberate    | Photographs, textures, fine details |
| RCAN         | 5             | Deliberate    | Rich textures, premium results      |
| FSRCNN       | 4             | Fast          | Balanced general usage              |
| ESPCN        | 3             | Lightning     | Previews, UI assets, simple images  |
| FSRCNN Small | 3             | Lightning     | Instant feedback, lightweight       |
| LapSRN       | 4             | Steady        | Line art, sharp edges               |

The "Auto (best quality)" option picks a high-quality model per scale (usually EDSR if present).

---

## Performance Notes

- OpenCV DNN Super Resolution typically runs on CPU by default. Expect slower performance for large outputs (for example, more than 40 megapixels).
- Luminara caches loaded models and performs in-memory encoding to keep the UI responsive.
- If processing is slow:
  - Try a smaller scale or a faster model (ESPCN or FSRCNN).
  - Close other heavy apps to free CPU and RAM.
  - Use WEBP or JPG for smaller output files.

---

## Troubleshooting

- "No trained models found." -> Ensure `.pb` files are placed under `models/<scale>/` as shown above.
- "OpenCV failed to load the model." -> The file may be corrupted or incompatible. Verify the path and file, or try a different model.
- "Could not decode the image." -> The input may be unsupported or damaged. Convert it to PNG or JPG and try again.
- Extremely slow on large images -> Use a faster model (ESPCN or FSRCNN) or a smaller scale; very high megapixel outputs are expensive on CPU.

---

## Project Structure

```
.
|-- models/                 # Pretrained .pb weights organized by scale
|-- streamlit_app.py        # Main Streamlit app
|-- requirements.txt        # Python dependencies
|-- readme.md               # You are here
`-- result.jpeg             # Optional demo image (if present)
```

Key modules in `streamlit_app.py`:

- Dynamic model registry and metadata mapping
- Cached model loading (cv2.dnn_superres) and inference
- Enhancement pipeline (sharpen, contrast/CLAHE, denoise)
- Split-view comparison and in-memory downloads
- Render history with thumbnails

---

## Development

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run streamlit_app.py
```

Code style: follow existing patterns, keep pull requests focused, and prefer clear, maintainable changes over cleverness.

---

<img width="2554" height="1271" alt="luminara1" src="https://github.com/user-attachments/assets/e727073a-1f22-4707-87b4-2149271692d8" />
<img width="2556" height="1263" alt="luminara2" src="https://github.com/user-attachments/assets/25dc43a3-4eed-4ce1-a099-36e17383bc7b" />

---

## Roadmap

- Optional integration of Real-ESRGAN for higher-end results (heavier dependency)
- Batch processing and queued jobs for multiple files
- Face-aware enhancements for portraits
- On-device GPU acceleration guidance and prebuilt binaries where feasible
- Theming and preset profiles (Photo, Art, UI)

---

## Contributing

Contributions are welcome. Please open an issue to discuss significant changes. When submitting pull requests, include a clear description, rationale, and testing notes (screenshots help).

---


## Acknowledgements

- OpenCV DNN Super Resolution (cv2.dnn_superres)
- EDSR, ESPCN, FSRCNN, LapSRN, and RCAN model authors and communities
- Streamlit for a delightful developer and user experience


