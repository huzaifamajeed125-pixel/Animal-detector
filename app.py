

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(
    page_title="Animal Classifier",
    page_icon="🐾",
    layout="centered"
)

st.title("🐾 Animal Classification")
st.write("Upload an image of a cat, dog, elephant, lion, or tiger.")

MODEL_PATH = Path(__file__).parent / "animal_classifier.keras"
CLASS_PATH = Path(__file__).parent / "class_names.txt"

@st.cache_resource
def load_animal_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": preprocess_input},
        compile=False
    )

@st.cache_data
def load_class_names():
    if CLASS_PATH.exists():
        names = CLASS_PATH.read_text().splitlines()
        if len(names) == 5:
            return names
    return ["cat", "dog", "elephant", "lion", "tiger"]

try:
    model = load_animal_model()
    class_names = load_class_names()
except Exception as e:
    st.error("Model load nahi ho saka. Files aur dependencies check karein.")
    st.exception(e)
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an animal image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    resized = image.resize((224, 224))
    image_array = np.array(resized, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    if st.button("Predict Animal", type="primary"):
        with st.spinner("Predicting..."):
            predictions = model.predict(image_array, verbose=0)[0]
            index = int(np.argmax(predictions))

        st.success(f"Predicted Animal: {class_names[index].title()}")
        st.metric(
            "Model Confidence",
            f"{predictions[index] * 100:.2f}%"
        )

        st.subheader("All Class Scores")
        for i, name in enumerate(class_names):
            st.write(f"{name.title()}: {predictions[i] * 100:.2f}%")
            st.progress(float(predictions[i]))
