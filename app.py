import os
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Parcel & Box Detector", page_icon="📦", layout="wide")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

st.title("📦 Parcel & Box Detection (Custom YOLOv8)")
st.write("Upload an image to detect cardboard **boxes** and **parcels** (poly-mailer bags).")

conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
st.sidebar.markdown(
    "**Classes:** Box, Parcel\n\n"
    "Lower the threshold to see more detections (and more false alarms)."
)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    result = model.predict(image, conf=conf, verbose=False)[0]

    annotated = result.plot()[:, :, ::-1]   # BGR -> RGB

    col1, col2 = st.columns(2)
    col1.subheader("Original")
    col1.image(image, use_container_width=True)
    col2.subheader("Detections")
    col2.image(annotated, use_container_width=True)

    names = [model.names[int(c)] for c in result.boxes.cls]
    m1, m2, m3 = st.columns(3)
    m1.metric("Boxes", names.count("Box"))
    m2.metric("Parcels", names.count("Parcel"))
    m3.metric("Total objects", len(names))

    if len(names) > 0:
        st.subheader("Detection details")
        rows = [
            {"Class": model.names[int(c)], "Confidence": round(float(p), 2)}
            for c, p in zip(result.boxes.cls, result.boxes.conf)
        ]
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No objects detected. Try lowering the confidence threshold.")

with st.expander("Known limitations"):
    st.write(
        "The model can miss large, close-up or partly cropped boxes and brown padded "
        "mailers, because the training data is mostly single boxes on plain backgrounds "
        "and white parcel bags on a conveyor belt."
    )