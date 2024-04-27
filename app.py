import numpy as np
import streamlit as st
import tensorflow as tf

def apply_sharpening(image_array, alpha=1.5):
    """
    Manually apply a sharpening filter to the image.
    `alpha` is the intensity of the sharpening.
    """
    # Define a sharpening kernel
    kernel = np.array([
        [-1, -1, -1],
        [-1, 9 + alpha, -1],
        [-1, -1, -1]
    ])

    # Padding the image to maintain dimensions
    pad_width = ((1, 1), (1, 1), (0, 0))  # Pad the spatial dimensions but not the channels
    image_padded = np.pad(image_array, pad_width, mode='constant', constant_values=0)

    # Image dimensions
    height, width, channels = image_padded.shape
    # Prepare an empty array for the output
    sharpened_image = np.zeros_like(image_array)

    # Apply the kernel to each pixel
    for y in range(1, height-1):
        for x in range(1, width-1):
            for c in range(channels):
                # Element-wise multiplication of the kernel and the pixel matrix
                region = image_padded[y-1:y+2, x-1:x+2, c]
                sharpened_value = np.sum(region * kernel)
                sharpened_image[y-1, x-1, c] = np.clip(sharpened_value, 0, 255)

    return sharpened_image

def preprocess_image(image, IMG_SIZE=(192, 192)):
    """
    Preprocess the image: resize, sharpen, and normalize.
    """
    # Resize the image to match the input shape expected by the model
    image = image.resize(IMG_SIZE)

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Manually sharpen the image
    sharpened_image = apply_sharpening(image_array)

    # Normalize pixel values to be between 0 and 1
    normalized_image = sharpened_image / 255.0

    # Expand dimensions to match model input shape
    processed_image = np.expand_dims(normalized_image, axis=0)
    
    return processed_image

def main():
    st.title('Lung Cancer Detection')

    # Add the image uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        from PIL import Image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Display the processed image (if needed for debugging)
        # st.image(processed_image[0], caption='Processed Image', use_column_width=True)

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

