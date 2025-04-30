import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load model once
model = tf.keras.models.load_model("model/brain_tumor_model5.h5")
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Configure Streamlit layout
st.set_page_config(layout="centered")
st.title("🧠 Brain Tumor Classifier - ITRI 616")
st.write("Upload an MRI scan to classify the tumor type.")

# Upload section
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI.', use_column_width=True)

    # Preprocess image
    img = image.resize((150, 150))  # Match training dimensions
    img_array = np.array(img) / 255.0
    img_expanded = np.expand_dims(img_array, axis=0)

    st.caption(f"Image normalized — pixel values range from `{img_array.min():.2f}` to `{img_array.max():.2f}`")


    # Predict
    prediction = model.predict(img_expanded)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    if np.max(prediction) < 0.6:
        st.warning("⚠️ Low confidence prediction – consider a medical expert review.")


    st.subheader("🧪 Prediction Result:")
    st.success(f"**Predicted Tumor Type:** `{predicted_class}`")
    st.info(f"**Confidence:** `{confidence * 100:.2f}%`")


    top_2 = prediction[0].argsort()[-2:][::-1]
    for idx in top_2:
        st.write(f"{class_names[idx]}: {prediction[0][idx]*100:.2f}%")


    # Probability chart
    st.subheader("📊 Prediction Probabilities:")
    df_probs = pd.DataFrame(prediction, columns=class_names).T
    df_probs.columns = ["Confidence"]
    st.bar_chart(df_probs)

# Divider
st.markdown("---")

# Show Training Performance Graphs
st.subheader("📈 Model Training Performance")

if os.path.exists("model/training_performance.png"):
    st.image("model/training_performance.png", caption="Training vs Validation Accuracy & Loss", use_column_width=True)
else:
    st.warning("Training performance plot not found.")

# Show Confusion Matrix
if os.path.exists("model/confusion_matrix_model6.png"):
    st.subheader("📉 Confusion Matrix")
    st.image("model/confusion_matrix_model6.png", caption="Model Confusion Matrix", use_column_width=True)
else:
    st.warning("Confusion matrix not found.")

# Final Stats (optional for summary)
st.markdown("---")
st.subheader("📋 Final Evaluation Summary (console values)")
st.markdown("""
- **Model File:** `brain_tumor_model5.h5`  
- **Test Accuracy:** `44.42%`  
- **Best Validation Accuracy:** `58.38%`  
- **Class Weights:** ❌ Not Applied  
- **Regularization:** L2 (0.0005)  
- **Learning Rate Reduction:** ✅ Enabled  
- **Early Stopping:** Patience = 7  
""")
