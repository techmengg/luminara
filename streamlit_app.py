import cv2
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from datetime import datetime

# Initialize the Super Resolution model
sr = cv2.dnn_superres.DnnSuperResImpl_create()

# Available models categorized by scale
models_2x = ['EDSR_x2.pb', 'ESPCN_x2.pb', 'FSRCNN-small_x2.pb', 'FSRCNN_x2.pb', 'LapSRN_x2.pb']
models_3x = ['EDSR_x3.pb', 'ESPCN_x3.pb', 'FSRCNN-small_x3.pb', 'FSRCNN_x3.pb']
models_4x = ['EDSR_x4.pb', 'ESPCN_x4.pb', 'FSRCNN-small_x4.pb', 'FSRCNN_x4.pb', 'LapSRN_x4.pb', 'rcan_x4.pb']
models_8x = ['LapSRN_x8.pb', 'rcan_x8.pb']

BASE_PATH = 'models/'
DEBUG = True  # Set to True for debugging logs

# Function to apply rescaling
def rescale_image(model_path: str, model_name: str, scale: str, img, img_type: str):
    """Rescales the given image using the selected AI model."""
    scale = int(scale.split('x')[0])

    # Debug: Ensure the model path exists
    if not model_path or not model_name:
        st.error(f"❌ Model path or name is invalid: {model_path}, {model_name}")
        return None, None

    # Load the model
    try:
        sr.readModel(model_path)
        sr.setModel(model_name, scale)
    except cv2.error as e:
        st.error(f"❌ OpenCV failed to load the model: {str(e)}")
        return None, None

    # Debug: Display model being used
    if DEBUG:
        st.write(f"🔍 Using model: {model_name}, Scale: {scale}")

    # Ensure the image is in the correct format
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Debug: Display the original image shape
    if DEBUG:
        st.write(f"📏 Original Image Shape: {img.shape}")

    # Apply rescaling
    result = sr.upsample(img_rgb)

    # Debug: Check if rescaling is actually modifying the image
    if DEBUG:
        st.write(f"📏 Rescaled Image Shape: {result.shape}")

    # Save and return result
    img_type = img_type.split('/')[1]
    save_path = f'result.{img_type}'
    plt.imsave(save_path, result)
    return result, save_path

# Function to return model type based on name
def get_modelname(selected_model: str) -> str:
    if 'EDSR' in selected_model:
        return 'edsr'
    elif 'LapSRN' in selected_model:
        return 'lapsrn'
    elif 'ESPCN' in selected_model:
        return 'espcn'
    elif 'FSRCNN' in selected_model:
        return 'fsrcnn'
    elif 'rcan' in selected_model.lower():
        return 'rcan'
    return ''

# Model selection function
def model_selector(scale: str) -> str:
    models = {'2x': models_2x, '3x': models_3x, '4x': models_4x, '8x': models_8x}
    if scale not in models:
        return False, False
    model = st.selectbox('Choose an AI model:', ['Not selected'] + models[scale])
    return model, get_modelname(model)

# Streamlit UI with updated Welcome Text
st.markdown("""
    <div style='text-align: center;'>
        <h1>✨ Welcome to Luminara! ✨</h1>
        <p>This AI-powered tool helps you <b>rescale images</b> using deep learning models.</p>
        <h4>📏 Scaling Guide:</h4>
        <p><b>2x, 3x:</b> Moderate rescaling, faster processing.</p>
        <p><b>4x, 8x:</b> Larger rescaling, recommended for detailed resizing.</p>
        <p>💡 <b>Tip:</b> If you're working with very low-quality images, try 4x before going to 8x.</p>
        <h4>🚀 How to Use:</h4>
        <p>1️⃣ Choose a scale factor (2x, 3x, 4x, or 8x).</p>
        <p>2️⃣ Select an AI model.</p>
        <p>3️⃣ Upload your image (JPG/PNG).</p>
        <p>4️⃣ Click <b>"Rescale Image"</b> and download the result.</p>
        <p>⚠️ <b>Note:</b> Higher scaling (8x) may be slow on Streamlit due to CPU limits.</p>
        <p>Developed by <a href='https://github.com/techmengg' target='_blank'>techmengg</a></p>
    </div>
""", unsafe_allow_html=True)

scale = st.selectbox('Select rescale factor:', ('Not selected', '2x', '3x', '4x', '8x'))

uploaded_file = None
model, model_name = model_selector(scale)
if model and model != 'Not selected':
    model_path = BASE_PATH + scale + '/' + model
    uploaded_file = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "png"])

image = None
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Show the uploaded image
    st.markdown("<h3 style='text-align: center;'>Uploaded Image</h3>", unsafe_allow_html=True)
    st.image(image, channels="BGR", use_column_width=True)

    # CPU limitation checks
    if scale == '8x' and image.shape[0] > 128:
        st.error("⚠️ Image is too large for 8x scaling due to CPU constraints.")
    elif scale == '4x' and image.shape[0] > 200:
        st.error("⚠️ Image is too large for 4x scaling due to CPU constraints.")
    elif scale == '3x' and image.shape[0] > 540:
        st.error("⚠️ Image is too large for 3x scaling due to CPU constraints.")
    elif scale == '2x' and image.shape[0] > 550:
        st.error("⚠️ Image is too large for 2x scaling due to CPU constraints.")
    else:
        if st.button('⚡ Rescale Image!'):
            st.info('⏳ Processing...')
            result, save_path = rescale_image(model_path, model_name, scale, image, uploaded_file.type)

            if result is not None:
                st.success('🎉 Image is ready for download!')
                st.balloons()

                # Show before/after images
                st.markdown("<h3 style='text-align: center;'>Before & After</h3>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original Image", channels="BGR", use_column_width=True)
                with col2:
                    st.image(result, caption="Rescaled Image", channels="RGB", use_column_width=True)

                # Download button
                with open(save_path, 'rb') as f:
                    st.download_button('⬇️ Download Image', f, file_name=f'{scale}_{datetime.now()}_{save_path}')
            else:
                st.error("❌ Rescaling failed. Check logs for errors.")
