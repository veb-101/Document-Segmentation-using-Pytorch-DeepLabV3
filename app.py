# Install CPU version of torch and torchvision on streamlit cloud
import os
import io
import gc
import cv2
import sys
import time
import subprocess
import numpy as np
import streamlit as st


try:
    import torch

# This block executes only on the first run when your package isn't installed
except ModuleNotFoundError as e:
    subprocess.Popen([f"{sys.executable} -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu"], shell=True)
    # wait for subprocess to install package before running your actual code below
    time.sleep(20)


# ------------------------------------------------------------
from torchvision.datasets.utils import download_file_from_google_drive

# Download trained models
if not os.path.exists(os.path.join(os.getcwd(), "model_mbv3_iou_mix_2C049.pth")):
    print("Downloading Deeplabv3 with MobilenetV3-Large backbone...")
    download_file_from_google_drive(file_id=r"1ROtCvke02aFT6wnK-DTAIKP5-8ppXE2a", root=os.getcwd(), filename=r"model_mbv3_iou_mix_2C049.pth")


if not os.path.exists(os.path.join(os.getcwd(), "model_r50_iou_mix_2C020.pth")):
    print("Downloading Deeplabv3 with ResNet-50 backbone...")
    download_file_from_google_drive(file_id=r"1DEl6qLckFChSDlT_oLUbO2JpN776Qx-g", root=os.getcwd(), filename=r"model_r50_iou_mix_2C020.pth")
# ------------------------------------------------------------


from utility_functions import load_model_DL_MBV3, load_model_DL_R50, get_image_download_link, deep_learning_scan, traditional_scan


def main(input_file, procedure, image_size=384):

    file_bytes = np.asarray(bytearray(input_file.read()), dtype=np.uint8)  # Read bytes
    image = cv2.imdecode(file_bytes, 1)[:, :, ::-1]  # Decode and convert to RGB

    st.write("Input image size:", image.shape)

    col1, col2 = st.columns((1, 1))

    with col1:
        st.title("Input")
        st.image(image, channels="RGB", use_column_width=True)

    with col2:
        st.title("Scanned")

        if procedure == "Traditional":
            output = traditional_scan(og_image=image)
        else:
            model = model_mbv3 if model_selected == "MobilenetV3-Large" else model_r50
            output = deep_learning_scan(og_image=image, trained_model=model, image_size=image_size)

        st.image(output, channels="RGB", use_column_width=True)

    return output


# Streamlit Components
st.set_page_config(
    page_title="Document Scanner | LearnOpenCV",
    page_icon="https://learnopencv.com/wp-content/uploads/2017/12/favicon.png",
    layout="centered",  # centered, wide
    initial_sidebar_state="expanded",
    menu_items={"About": "### Visit www.learnopencv.com for more exciting tutorials!!!",},
)

IMAGE_SIZE = 384
model_mbv3 = load_model_DL_MBV3(img_size=IMAGE_SIZE)
model_r50 = load_model_DL_R50(img_size=IMAGE_SIZE)

st.title("Document Scanner")

procedure_selected = st.radio("Select Scanning Procedure:", ("Traditional", "Deep Learning"), horizontal=True)

if procedure_selected == "Deep Learning":
    model_selected = st.radio("Select Document Segmentation Backbone Model:", ("MobilenetV3-Large", "ResNet-50"), horizontal=True)


tab1, tab2 = st.tabs(["Upload a Document", "Capture Document"])

with tab1:
    file_upload = st.file_uploader("Upload Document Image :", type=["jpg", "jpeg", "png"])
    output = None

    if file_upload is not None:
        output = main(input_file=file_upload, procedure=procedure_selected, image_size=IMAGE_SIZE)

        if output is not None:
            # buffered = save_image(scanned_output=output, format="PNG")
            # st.download_button(label="Download Scanned image", data=buffered, file_name=f"scanned_{file_upload.name}")
            st.markdown(get_image_download_link(output, f"scanned_{file_upload.name}", "Download scanned File"), unsafe_allow_html=True)


with tab2:
    output = None
    run = st.checkbox("Start Camera")

    if run:
        file_upload = st.camera_input("Capture Document", disabled=not run)
        if file_upload is not None:
            output = main(input_file=file_upload, procedure=procedure_selected, image_size=IMAGE_SIZE)

            if output is not None:
                # buffered = save_image(scanned_output=output, format="PNG")
                # st.download_button(label="Download Scanned image", data=buffered, file_name=f"scanned_{file_upload.name}")
                st.markdown(get_image_download_link(output, f"scanned_{file_upload.name}", "Download scanned File"), unsafe_allow_html=True)
