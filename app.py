# Install CPU version of torch and torchvision on streamlit cloud
import os
import gc
import cv2
import sys
import time
import subprocess
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas


try:
    import torch

# This block executes only on the first run when your package isn't installed
except ModuleNotFoundError as e:
    subprocess.Popen([f"{sys.executable} -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu"], shell=True)
    # wait for subprocess to install package before running your actual code below
    time.sleep(20)

import torch
import torchvision.transforms as torchvision_T
from torchvision.datasets.utils import download_file_from_google_drive
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50

# ------------------------------------------------------------
# Download trained models
if not os.path.exists(os.path.join(os.getcwd(), "model_mbv3_iou_mix_2C049.pth")):
    print("Downloading Deeplabv3 with MobilenetV3-Large backbone...")
    download_file_from_google_drive(file_id=r"1ROtCvke02aFT6wnK-DTAIKP5-8ppXE2a", root=os.getcwd(), filename=r"model_mbv3_iou_mix_2C049.pth")


if not os.path.exists(os.path.join(os.getcwd(), "model_r50_iou_mix_2C020.pth")):
    print("Downloading Deeplabv3 with ResNet-50 backbone...")
    download_file_from_google_drive(file_id=r"1DEl6qLckFChSDlT_oLUbO2JpN776Qx-g", root=os.getcwd(), filename=r"model_r50_iou_mix_2C020.pth")
# ------------------------------------------------------------


@st.cache(allow_output_mutation=True)
def load_model_DL_MBV3(num_classes=2, device=torch.device("cpu"), img_size=384):
    checkpoint_path = os.path.join(os.getcwd(), "model_mbv3_iou_mix_2C049.pth")
    checkpoints = torch.load(checkpoint_path, map_location=device)

    model = deeplabv3_mobilenet_v3_large(num_classes=num_classes, aux_loss=True).to(device)
    model.load_state_dict(checkpoints, strict=False)
    model.eval()
    _ = model(torch.randn((1, 3, img_size, img_size)))
    return model


@st.cache(allow_output_mutation=True)
def load_model_DL_R50(num_classes=2, device=torch.device("cpu"), img_size=384):
    checkpoint_path = os.path.join(os.getcwd(), "model_r50_iou_mix_2C020.pth")
    checkpoints = torch.load(checkpoint_path, map_location=device)

    model = deeplabv3_resnet50(num_classes=num_classes, aux_loss=True).to(device)
    model.load_state_dict(checkpoints, strict=False)
    model.eval()
    _ = model(torch.randn((1, 3, img_size, img_size)))
    return model


def image_preprocess_transforms(mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)):
    common_transforms = torchvision_T.Compose([torchvision_T.ToTensor(), torchvision_T.Normalize(mean, std),])
    return common_transforms


def order_points(pts):
    """Rearrange coordinates to order:
    top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect.astype("int").tolist()


def find_dest(pts):
    (tl, tr, br, bl) = pts
    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Final destination co-ordinates.
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]

    return order_points(destination_corners)


def scan(image_true=None, trained_model=None, image_size=384, BUFFER=10, preprocess_transforms=image_preprocess_transforms()):
    IMAGE_SIZE = image_size
    half = IMAGE_SIZE // 2

    imH, imW, C = image_true.shape

    image_model = cv2.resize(image_true, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    scale_x = imW / IMAGE_SIZE
    scale_y = imH / IMAGE_SIZE

    image_model = preprocess_transforms(image_model)
    image_model = torch.unsqueeze(image_model, dim=0)

    with torch.no_grad():
        out = trained_model(image_model)["out"].cpu()

    del image_model
    gc.collect()

    out = torch.argmax(out, dim=1, keepdims=True).permute(0, 2, 3, 1)[0].numpy().squeeze().astype(np.int32)
    r_H, r_W = out.shape

    _out_extended = np.zeros((IMAGE_SIZE + r_H, IMAGE_SIZE + r_W), dtype=out.dtype)
    _out_extended[half : half + IMAGE_SIZE, half : half + IMAGE_SIZE] = out * 255
    out = _out_extended.copy()

    del _out_extended
    gc.collect()

    # Edge Detection.
    canny = cv2.Canny(out.astype(np.uint8), 225, 255)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    page = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # ==========================================
    epsilon = 0.02 * cv2.arcLength(page, True)
    corners = cv2.approxPolyDP(page, epsilon, True)

    corners = np.concatenate(corners).astype(np.float32)

    corners[:, 0] -= half
    corners[:, 1] -= half

    corners[:, 0] *= scale_x
    corners[:, 1] *= scale_y

    # check if corners are inside.
    # if not find smallest enclosing box, expand_image then extract document
    # else extract document

    if not (np.all(corners.min(axis=0) >= (0, 0)) and np.all(corners.max(axis=0) <= (imW, imH))):

        left_pad, top_pad, right_pad, bottom_pad = 0, 0, 0, 0

        rect = cv2.minAreaRect(corners.reshape((-1, 1, 2)))
        box = cv2.boxPoints(rect)
        box_corners = np.int32(box)
        # box_corners = minimum_bounding_rectangle(corners)

        box_x_min = np.min(box_corners[:, 0])
        box_x_max = np.max(box_corners[:, 0])
        box_y_min = np.min(box_corners[:, 1])
        box_y_max = np.max(box_corners[:, 1])

        # Find corner point which doesn't satify the image constraint
        # and record the amount of shift required to make the box
        # corner satisfy the constraint
        if box_x_min <= 0:
            left_pad = abs(box_x_min) + BUFFER

        if box_x_max >= imW:
            right_pad = (box_x_max - imW) + BUFFER

        if box_y_min <= 0:
            top_pad = abs(box_y_min) + BUFFER

        if box_y_max >= imH:
            bottom_pad = (box_y_max - imH) + BUFFER

        # new image with additional zeros pixels
        image_extended = np.zeros((top_pad + bottom_pad + imH, left_pad + right_pad + imW, C), dtype=image_true.dtype)

        # adjust original image within the new 'image_extended'
        image_extended[top_pad : top_pad + imH, left_pad : left_pad + imW, :] = image_true
        image_extended = image_extended.astype(np.float32)

        # shifting 'box_corners' the required amount
        box_corners[:, 0] += left_pad
        box_corners[:, 1] += top_pad

        corners = box_corners
        image_true = image_extended

    corners = sorted(corners.tolist())
    corners = order_points(corners)
    destination_corners = find_dest(corners)
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))

    final = cv2.warpPerspective(image_true, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LANCZOS4)
    final = np.clip(final, a_min=0, a_max=255)
    final = final.astype(np.uint8)

    return final


def main(image, model=None):
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)  # Read bytes
    cv_image = cv2.imdecode(file_bytes, 1)[:, :, ::-1]  # Decode and convert to RGB

    col1, col2 = st.columns((1, 1))

    with col1:
        st.title("Input")
        st.image(cv_image, channels="RGB", use_column_width=True)

    with col2:
        st.title("Scanned")
        output = scan(image_true=cv_image, trained_model=model, image_size=IMAGE_SIZE)
        st.image(output, channels="RGB", use_column_width=True)

    # Download scanned file
    if output is not None:
        cv2.imwrite(file_save_path, output[:, :, ::-1])
        with open(file_save_path, "rb") as file:
            st.download_button(label="Download Scanned image", data=file, file_name=f"scanned_{file_upload.name}")

    return output


# We create a downloads directory within the streamlit static asset directory
# and we write output files to it
file_save_path = os.path.join(st.__path__[0], "static", "downloads", "output.png")
os.makedirs(os.path.dirname(file_save_path), exist_ok=True)

IMAGE_SIZE = 384

# Streamlit Components
st.set_page_config(
    page_title="Document Segmentation using Pytorch | LearnOpenCV",
    page_icon="https://learnopencv.com/wp-content/uploads/2017/12/favicon.png",
    layout="centered",  # centered, wide
    initial_sidebar_state="expanded",
    menu_items={"About": "### Visit www.learnopencv.com for more exciting tutorials!!!",},
)

model_mbv3 = load_model_DL_MBV3(img_size=IMAGE_SIZE)
model_r50 = load_model_DL_R50(img_size=IMAGE_SIZE)

st.title("Document Scanner: Semantic Segmentation using DeepLabV3-PyTorch")

select_model = st.radio("Select Document Segmentation Model:", ("MobilenetV3-Large", "Resnet-50"), horizontal=True)
model = model_mbv3 if select_model == "MobilenetV3-Large" else model_r50

tab1, tab2 = st.tabs(["Upload a Document", "Capture Document"])

with tab1:
    # with st.form("my-form", clear_on_submit=True):
    #     file_upload = st.file_uploader("Upload Document Image :", type=["jpg", "jpeg", "png"])
    #     submitted = st.form_submit_button("Scan!")
    # if submitted and file_upload is not None:
    #     output = main(file_upload, model=model)

    file_upload = st.file_uploader("Upload Document Image :", type=["jpg", "jpeg", "png"])
    if file_upload:
        _ = main(file_upload, model=model)

with tab2:
    run = st.checkbox("Start Camera")

    if run:
        file_upload = st.camera_input("Capture Document", disabled=not run)
        if file_upload:
            _ = main(file_upload, model=model)
