import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

model = load_model('C:\\Users\\HP\\Downloads\\khalid\\lungModel2.h5')

def preprocess_image(image):
    # Convert the image to a numpy array
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    
    # Expand the dimensions of the image to match the input shape of the model
    image_array = tf.expand_dims(image_array, axis=0)
    
    # Define the parameters for data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Apply data augmentation to the image
    augmented_image = next(datagen.flow(image_array, batch_size=1))
    
    return augmented_image

def main():
    st.title('Lung Cancer Detection')
    st.write('Upload an image of a lung for cancer detection')

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make predictions
        prediction = model.predict(augmented_image)
        classes = ['normal', 'adenocarcinoma', 'large.cell', 'squamous']
        predicted_class = classes[np.argmax(prediction)]

        # Display the result
        st.write('Prediction:', predicted_class)

