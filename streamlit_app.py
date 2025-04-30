import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

#Load model once
model = tf.keras.models.load_model("model/brain_tumor_model.h5")
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

st.title("Brain Tumor Classifier - ITRI 616")
st.write("Upload an MRI scan to classify the tumor type")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI.', use_column_width=True)

    #Preprocess image
    img = image.resize((128, 128))
    img = np.expand_dims(np.array(img) / 255.0, axis=0)

    #Predict
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"Confidence: {np.max(prediction) * 100:.2f}%")
