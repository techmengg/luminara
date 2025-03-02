# Luminara: AI-Powered Image Upscaler (Open-Source)

## Overview
Luminara is an AI-powered image rescaling tool that utilizes deep learning models to upscale images while preserving details. It provides a simple and interactive web interface using Streamlit and supports multiple upscaling factors (2x, 3x, 4x, and 8x).

## Features
- **AI-based Super Resolution**: Uses deep learning models for high-quality image upscaling.
- **Multiple Scaling Options**: Supports **2x, 3x, 4x, and 8x** scaling factors.
- **Easy-to-Use UI**: Built with Streamlit for a smooth and intuitive experience.
- **Model Selection**: Choose from different AI models for the best upscaling results.
- **Supports JPG & PNG**: Works with widely used image formats.
- **Open Source**: Licensed under the MIT License.

## Installation

To use Luminara locally, follow these steps:

```bash
# Clone the repository
git clone https://github.com/your-username/Luminara.git
cd Luminara

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:

```bash
python3 -m streamlit run streamlit_app.py
```

Then open `http://localhost:8501/` in your web browser.

## Available AI Models
Luminara supports the following deep learning models for super-resolution:

| Scale Factor | Models Available |
|-------------|-----------------|
| **2x** | EDSR, ESPCN, FSRCNN, LapSRN |
| **3x** | EDSR, ESPCN, FSRCNN |
| **4x** | EDSR, ESPCN, FSRCNN, LapSRN, RCAN |
| **8x** | LapSRN, RCAN |

## How It Works
### 1️⃣ Select Rescale Factor
Choose from **2x, 3x, 4x, or 8x** scaling options.

### 2️⃣ Choose an AI Model
Select the model that best fits your image enhancement needs.

### 3️⃣ Upload an Image
Supports **JPG and PNG** formats.

### 4️⃣ Process & Download
Click "Rescale Image" to apply the AI model and download the enhanced image.

![home](https://github.com/user-attachments/assets/cc95342a-fca6-4f5c-b1b8-8feef6583be2)
![results](https://github.com/user-attachments/assets/0f1cdd84-8d24-423b-8da1-af55ae73241c)


## 🛠 Code Snippets
### Load and Apply Super-Resolution Model
```python
import cv2
import numpy as np

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel('models/EDSR_x4.pb')
sr.setModel('edsr', 4)

img = cv2.imread('input.jpg')
upscaled = sr.upsample(img)
cv2.imwrite('output.jpg', upscaled)
```

### Streamlit UI Code
```python
import streamlit as st
st.title("Luminara: AI Image Upscaler")
scale = st.selectbox("Select Rescale Factor", ['2x', '3x', '4x', '8x'])
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png'])
```

## ⚠️ Limitations
- **Processing Time**: Higher scaling factors (especially 8x) may take longer.
- **Resource Intensive**: Running on a CPU may be slow; a GPU is recommended.
- **Model Compatibility**: RCAN models may not work outside of OpenCV.

---

> Developed by **techmengg** | [GitHub](https://github.com/techmengg)



