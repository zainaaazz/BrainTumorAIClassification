# 🧠 Brain-MRI Tumour Classifier (Streamlit demo)

A lightweight Streamlit front-end that loads the **DenseNet-121** model from  
**`22-05-25_attempt2.h5`** (iteration 15 of my project) and predicts one of
four tumour classes from a single MRI slice:

| Class label | Folder name |
|-------------|-------------|
| Glioma tumour | `glioma_tumor` |
| Meningioma tumour | `meningioma_tumor` |
| *No tumour* | `no_tumor` |
| Pituitary tumour | `pituitary_tumor` |

<kbd>Upload → Predict → View confidence bar chart</kbd>

---

## 📁 Project structure
.
├─ app.py # Streamlit app
├─ 22-05-25_attempt2.h5 # Saved Keras model (HDF5)
└─ README.md # (this file)


---

## 🚀 Quick-start

> Tested with **Python ≥ 3.9**  
> CPU-only TensorFlow works fine for inference.

```bash
# 1) (optional) create & activate a virtual-env
python -m venv venv
# On Windows:  venv\Scripts\activate
source venv/bin/activate

# 2) install dependencies
pip install streamlit tensorflow pillow matplotlib

# macOS Apple-silicon alt:
# pip install tensorflow-macos tensorflow-metal streamlit pillow matplotlib

# 3) run the app
streamlit run app.py

A browser tab will open at http://localhost:8501.
Upload a PNG/JPG MRI slice and inspect the prediction.

# How it works (high-level)
Model load – tf.keras.models.load_model('22-05-25_attempt2.h5')

Image pre-processing – resized to 224 × 224, preprocess_input() from
Keras DenseNet family.

Prediction – Soft-max vector (length 4).
Highest value → predicted class.

Display – Streamlit shows:
-the uploaded image
-predicted class + confidence %
-bar chart of the soft-max probabilities





