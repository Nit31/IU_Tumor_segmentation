import streamlit as st
import requests
import base64
import numpy as np
import cv2

# Set the title of the Streamlit app
st.title("Tumor Segmentation")

# Add a file uploader widget for uploading images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Initialize session state to hold the output image and label
if 'output_image' not in st.session_state:
    st.session_state.output_image = None
if 'label' not in st.session_state:
    st.session_state.label = None

# If a file is uploaded
if uploaded_file:
    # The URL of the FastAPI endpoint
    url = 'http://api:3010/predict/'

    # Send the image to FastAPI
    files = {'file': uploaded_file}
    response = requests.post(url, files=files)

    if response.status_code == 200:
        data = response.json()
        overlay_image = base64.b64decode(data['overlay'])
        nparr = np.frombuffer(overlay_image, np.uint8)
        output_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Convert to a usable image format

        # Store the output image and label in session state
        st.session_state.output_image = output_image

    else:
        st.error("Error: Could not retrieve the masked image.")

# Display the original image
if uploaded_file:
    st.image(uploaded_file, caption='Original Image', use_column_width=True)

# Display the output image with the mask overlay if it exists
if st.session_state.output_image is not None:
    st.image(st.session_state.output_image, channels="BGR", caption='Image with Highlight', use_column_width=True)
