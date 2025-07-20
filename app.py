import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('cats_vs_dogs_model.h5')

st.title("Cat vs Dog Classifier")
st.write("Upload an image to classify it as a cat or dog.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the class
    predictions = model.predict(img_array)
    confidence = predictions[0][0]

    if confidence > 0.5:
        st.write(f"It's a Dog with {(confidence * 100):.2f}% confidence.")
    else:
        st.write(f"It's a Cat with {((1 - confidence) * 100):.2f}% confidence.")
