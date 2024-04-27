import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Dummy user database
users = {
    "admin": "password123",
    "user": "testpass"
}

def apply_sharpening(image_array, alpha=0.3):
    kernel = np.array([
        [-1, -1, -1],
        [-1, 9 + alpha, -1],
        [-1, -1, -1]
    ])
    pad_width = ((1, 1), (1, 1), (0, 0))
    image_padded = np.pad(image_array, pad_width, mode='constant', constant_values=0)
    height, width, channels = image_padded.shape
    sharpened_image = np.zeros_like(image_array)
    for y in range(1, height-1):
        for x in range(1, width-1):
            for c in range(channels):
                region = image_padded[y-1:y+2, x-1:x+2, c]
                sharpened_value = np.sum(region * kernel)
                sharpened_image[y-1, x-1, c] = np.clip(sharpened_value, 0, 255)
    return sharpened_image

def preprocess_image(image, IMG_SIZE=(192, 192)):
    image = image.resize(IMG_SIZE)
    image_array = np.array(image)
    sharpened_image = apply_sharpening(image_array)
    normalized_image = sharpened_image / 255.0
    processed_image = np.expand_dims(normalized_image, axis=0)
    return processed_image

def login_page():
    st.title("Login Page")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if username in users and users[username] == password:
            st.session_state['user'] = username
            st.session_state['page'] = 'model'
        else:
            st.error("Invalid username or password")

def model_page():
    st.title("Lung Cancer Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        processed_image = preprocess_image(image)
        st.image(processed_image[0], caption='Processed Image', use_column_width=True)
        model_path = 'lungModel2.h5'
        model = tf.keras.models.load_model(model_path)
        prediction = model.predict(processed_image)
        classes = ['normal', 'adenocarcinoma', 'large.cell', 'squamous']
        predicted_class = classes[np.argmax(prediction)]
        st.write('Prediction:', predicted_class)
        if st.button("Provide Feedback"):
            st.session_state['page'] = 'feedback'

def feedback_page():
    st.title("Feedback")
    feedback = st.text_area("How was your experience?")
    if st.button("Submit Feedback"):
        # Here you could write the feedback to a file or database
        st.success("Thank you for your feedback!")
        st.session_state['page'] = 'model'

def main():
    if 'page' not in st.session_state:
        st.session_state['page'] = 'login'

    if 'user' not in st.session_state:
        st.session_state['user'] = None

    if st.session_state['page'] == 'login':
        login_page()
    elif st.session_state['page'] == 'model':
        model_page()
    elif st.session_state['page'] == 'feedback':
        feedback_page()

if __name__ == "__main__":
    main()
