import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import tensorflow.keras.applications.mobilenet as mobilenet
from scipy import ndimage

# Load the Keras model using TensorFlow's load_model method
@st.cache(allow_output_mutation=True)
def load_keras_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_image(image):
    # Resize the image to match the input shape expected by MobileNet
    IMG_SIZE = (224, 224)
    resized_image = image.resize(IMG_SIZE)

    # Convert the resized image to array
    resized_image = np.array(resized_image)

    # Sharpen the image for enhanced details
    sharpened_image = ndimage.median_filter(resized_image, 3)
    sharpened_image = resized_image + 0.8 * (resized_image - sharpened_image)

    # Preprocess the image using MobileNet's preprocess_input function
    processed_image = mobilenet.preprocess_input(sharpened_image)

    # Expand dimensions to match model input shape
    processed_image = np.expand_dims(processed_image, axis=0)
    
    return processed_image

def main():
    st.title('Lung Cancer Detection')
    
    # Add a login section
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username == "admin" and password == "123":
            st.sidebar.success("Logged in as {}".format(username))
        else:
            st.sidebar.error("Incorrect username or password")
            st.stop()

    st.write('Upload an image of a lung for cancer detection')

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Load the Keras model
        model_path = 'lungModel2.h5'
        model = load_keras_model(model_path)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make predictions
        prediction = model.predict(processed_image)
        classes = ['normal', 'adenocarcinoma', 'large.cell', 'squamous']
        predicted_class = classes[np.argmax(prediction)]

        # Display the result
        st.write('Prediction:', predicted_class)

        # Add a feedback section
        f
