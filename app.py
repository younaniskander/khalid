import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

# Load the Keras model
@st.cache(allow_output_mutation=True)
def load_keras_model(model_path):
    return load_model(model_path)

def preprocess_image(image):
    # Resize the image to match the input shape expected by the model
    IMG_SIZE = (192, 192)
    resized_image = image.resize(IMG_SIZE)

    # Convert the resized image to array
    resized_image = np.array(resized_image) / 255.0

    # Expand dimensions to match model input shape
    resized_image = np.expand_dims(resized_image, axis=0)
    
    return resized_image

def main():
    st.title('Lung Cancer Detection')
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

if __name__ == "__main__":
    main()
