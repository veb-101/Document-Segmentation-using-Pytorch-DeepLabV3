import base64
import io
import gc
import cv2
import PIL
import time
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas


import torch
import torchvision.transforms as torchvision_T

# Generating a link to download a particular image file.
# @st.cache(allow_output_mutation=True)
def get_image_download_link(img, filename, text):
    with st.spinner("Generating download link"):
        img = PIL.Image.fromarray(img)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
        time.sleep(2)
    return href


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


def image_preprocess_transforms(mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)):
    common_transforms = torchvision_T.Compose([torchvision_T.ToTensor(), torchvision_T.Normalize(mean, std),])
    return common_transforms


def generate_output(image: np.array, corners: list, scale: tuple = None, resize_shape: int = 640):
    corners = order_points(corners)

    if scale is not None:
        print(np.array(corners).shape, scale)
        corners = np.multiply(corners, scale)

    destination_corners = find_dest(corners)
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    out = cv2.warpPerspective(image, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LANCZOS4)
    out = np.clip(out, a_min=0, a_max=255)
    out = out.astype(np.uint8)
    return out


def traditional_scan(og_image: np.array):
    # Resize image to workable size
    dim_limit = 1080
    max_dim = max(og_image.shape)
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        og_image = cv2.resize(og_image, None, fx=resize_scale, fy=resize_scale)
    # Create a copy of resized original image for later use
    orig_img = og_image.copy()
    # Repeated Closing operation to remove text from the document.
    kernel = np.ones((5, 5), np.uint8)
    og_image = cv2.morphologyEx(og_image, cv2.MORPH_CLOSE, kernel, iterations=3)
    # GrabCut
    mask = np.zeros(og_image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (20, 20, og_image.shape[1] - 20, og_image.shape[0] - 20)
    cv2.grabCut(og_image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    og_image = og_image * mask2[:, :, np.newaxis]

    gray = cv2.cvtColor(og_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    # Edge Detection.
    canny = cv2.Canny(gray, 0, 200)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Finding contours for the detected edges.
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Keeping only the largest detected contour.
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Detecting Edges through Contour approximation.
    # Loop over the contours.
    if len(page) == 0:
        return orig_img
    for c in page:
        # Approximate the contour.
        epsilon = 0.02 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, epsilon, True)
        # If our approximated contour has four points.
        if len(corners) == 4:
            break
    # Sorting the corners and converting them to desired shape.
    corners = sorted(np.concatenate(corners).tolist())
    output = generate_output(orig_img, corners)

    return output


def deep_learning_scan(og_image: np.array = None, trained_model=None, image_size=384, BUFFER=10, preprocess_transforms=image_preprocess_transforms()):
    half = image_size // 2

    imH, imW, C = og_image.shape

    image_model = cv2.resize(og_image, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    scale_x = imW / image_size
    scale_y = imH / image_size

    image_model = preprocess_transforms(image_model)
    image_model = torch.unsqueeze(image_model, dim=0)

    with torch.no_grad():
        out = trained_model(image_model)["out"]

    del image_model
    gc.collect()

    out = torch.argmax(out, dim=1, keepdims=True).permute(0, 2, 3, 1)[0].numpy().squeeze().astype(np.int32)
    r_H, r_W = out.shape

    _out_extended = np.zeros((image_size + r_H, image_size + r_W), dtype=out.dtype)
    _out_extended[half : half + image_size, half : half + image_size] = out * 255
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
        image_extended = np.zeros((top_pad + bottom_pad + imH, left_pad + right_pad + imW, C), dtype=og_image.dtype)

        # adjust original image within the new 'image_extended'
        image_extended[top_pad : top_pad + imH, left_pad : left_pad + imW, :] = og_image
        image_extended = image_extended.astype(np.float32)

        # shifting 'box_corners' the required amount
        box_corners[:, 0] += left_pad
        box_corners[:, 1] += top_pad

        corners = box_corners
        og_image = image_extended

    corners = sorted(corners.tolist())
    output = generate_output(og_image, corners)

    return output


# @st.cache(allow_output_mutation=True)
# def save_image(scanned_output: np.array, format: str = "PNG"):
#     buffered = io.BytesIO()
#     PIL.Image.fromarray(scanned_output).save(buffered, format=format)
#     time.sleep(2)
#     return buffered


def aspect_ratio_resize(image_h, image_w, resize_to=400):
    asp = image_w / image_h

    if image_h > image_w:
        new_h = resize_to
        new_w = asp * new_h

    else:
        new_w = resize_to
        new_h = new_w / asp

    return int(round(new_h)), int(round(new_w))


def manual_scan(og_image: np.array, resize_shape=640):

    image_h, image_w, _ = og_image.shape
    asp_h, asp_w = aspect_ratio_resize(image_h, image_w, resize_to=resize_shape)

    scale_h = image_h / asp_h
    scale_w = image_w / asp_w

    st.markdown("###### Select 4 corners points.")
    st.markdown(
        """
        ###### Steps
        <ul>
        <li>Left-click to begin.</li>
        <li>Right-click when selecting the last point.</li>
        <li>Double-click to undo last selected point.</li>
        </ul>
        (On mouse pads, click instead of taps.)<br><br>

        """,
        unsafe_allow_html=True,
    )

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=3,
        background_image=PIL.Image.fromarray(og_image).resize((asp_h, asp_w)),
        update_streamlit=True,
        height=asp_h,
        width=asp_w,
        drawing_mode="polygon",
        key="canvas",
    )
    st.caption("Happy with the manual selection?")

    if st.button("Get Scanned"):
        # Get corner points
        corners = [i[1:3] for i in canvas_result.json_data["objects"][0]["path"][:4]]

        # Generate output
        final = generate_output(og_image, corners, scale=(scale_h, scale_w), resize_shape=resize_shape)
        st.image(final, channels="RGB", use_column_width=True)

        return final

