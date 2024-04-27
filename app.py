import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import io

def main():
    st.title('Lung Cancer Detection')

    # Add the image uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Convert the processed image back to a displayable format
        display_image = Image.fromarray((processed_image[0] * 255).astype(np.uint8))

        # Display the processed image
        st.image(display_image, caption='Processed Image', use_column_width=True)

        # Save the processed image to a temporary buffer
        buf = io.BytesIO()
        display_image.save(buf, format='PNG')
        byte_im = buf.getvalue()

        # Create a download button for the processed image
        st.download_button(
            label="Download Processed Image",
            data=byte_im,
            file_name="processed_image.png",
            mime="image/png"
        )

        # Load the Keras model
        model_path = 'lungModel2.h5'
        model = tf.keras.models.load_model(model_path)

        # Make predictions
        prediction = model.predict(processed_image)
        classes = ['normal', 'adenocarcinoma', 'large.cell', 'squamous']
        predicted_class = classes[np.argmax(prediction)]

        st.write('Prediction:', predicted_class)

if __name__ == "__main__":
    main()
