import cv2
import easyocr
import keras_ocr
import matplotlib.pyplot as plt
import numpy as np
import pytesseract as tess
import requests
import streamlit as st


def main():
    st.title("Testing OCR libraries on code snippets")

    picture_url = st.text_input(
        "Enter Tweet picture ID",
        value="https://pbs.twimg.com/media/FCIX6IUWQAgpiPT?format=jpg&name=large",
    )
    if picture_url == "":
        st.info("Please enter an ID")
        st.stop()

    with st.sidebar:
        st.header("Configuration")
        select_tesseract = st.checkbox("Compute tesseract")
        select_keras = st.checkbox("Compute keras-ocr")
        select_easyocr = st.checkbox("Compute easyocr")

    if select_tesseract:
        compute_tesseract(picture_url)
    if select_keras:
        compute_keras(picture_url)
    if select_easyocr:
        compute_easyocr(picture_url)


def compute_tesseract(picture_url: str):
    """
    https://www.opcito.com/blogs/extracting-text-from-images-with-tesseract-ocr-opencv-and-python
    https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html
    """
    r = requests.get(picture_url, stream=True).raw
    raw_image_data = np.asarray(bytearray(r.read()), dtype="uint8")
    image = cv2.imdecode(raw_image_data, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]
    inverted_img = cv2.bitwise_not(threshold_img)

    def _compute_ocr(label, img):
        # Engine Mode (--oem)
        # OCR engine mode 	Working description
        # 0 	Legacy engine only
        # 1 	Neural net LSTM only
        # 2 	Legacy + LSTM mode only
        # 3 	By Default, based on what is currently available
        custom_oem_psm_config = r"--oem 3 --psm 6"
        text = tess.image_to_string(img, config=custom_oem_psm_config)
        st.subheader(label)
        c1, c2 = st.columns((1, 2))
        c1.image(img)
        c2.code(text)

    _compute_ocr("BGR image", image)
    _compute_ocr("RGB image", rgb_image)
    _compute_ocr("Binary image", threshold_img)
    _compute_ocr("Inverted image", inverted_img)


def compute_keras(picture_url):
    """
    Keras-ocr
    Keras CRNN (text recognizer https://github.com/janzd/CRNN)
    CRAFT (text detector https://github.com/clovaai/CRAFT-pytorch)
    """
    pipeline = keras_ocr.pipeline.Pipeline()

    images = [keras_ocr.tools.read(url) for url in [picture_url]]
    prediction_groups = pipeline.recognize(images)

    fig, ax = plt.subplots(figsize=(20, 20))
    keras_ocr.tools.drawAnnotations(
        image=images[0], predictions=prediction_groups[0], ax=ax
    )

    st.pyplot(fig)


def compute_easyocr(picture_url: str):
    r = requests.get(picture_url, stream=True).raw
    raw_image_data = np.asarray(bytearray(r.read()), dtype="uint8")
    image = cv2.imdecode(raw_image_data, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]
    inverted_img = cv2.bitwise_not(threshold_img)

    reader = easyocr.Reader(["en"], gpu=False)
    resp = reader.readtext(inverted_img, detail=0, paragraph=False)

    st.subheader("EasyOCR")
