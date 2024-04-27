import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from streamlit_for_flutter import build_api

# Load the AI model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('lung_model2.h5')
    return model

model = load_model()

# Function to preprocess image
def preprocess_image(image):
    # Your preprocessing code here
    return processed_image

# Streamlit app
st.title('Lung Cancer Detection')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions
    prediction = model.predict(processed_image)
    classes = ['normal', 'adenocarcinoma', 'large.cell', 'squamous']
    predicted_class = classes[np.argmax(prediction)]

    st.write('Prediction:', predicted_class)

# Build the API
api_button_clicked = st.button("Generate API")

if api_button_clicked:
    api_url = build_api()
    st.write("API Endpoint:", api_url)
