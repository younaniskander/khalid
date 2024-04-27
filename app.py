import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import base64
import io
import pandas as pd

# Dummy user database
users = {
    "admin": "password123",
    "user": "testpass"
}

def preprocess_image(image, IMG_SIZE=(192, 192)):
    image = image.resize(IMG_SIZE)
    image_array = np.array(image)
    normalized_image = image_array / 255.0
    processed_image = np.expand_dims(normalized_image, axis=0)
    return processed_image

def login_page():
    st.title("Login Page")
    
    # Add image
    st.image("happy.jfif", caption='Your Image Caption', use_column_width=True)
    
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
        name = st.text_input("Name", "")
        gender = st.radio("Gender", ("Male", "Female"))
        age = st.number_input("Age", min_value=0, max_value=150, value=30)
        smoking_status = st.selectbox("Do you smoke?", ("Yes", "No"))

        image = Image.open(uploaded_file)
        st.image(image, caption='Original Image', use_column_width=True)
        processed_image = preprocess_image(image)
        st.image(processed_image[0], caption='Processed Image', use_column_width=True)
        model_path = 'lungModel2.h5'
        model = tf.keras.models.load_model(model_path)
        prediction = model.predict(processed_image)
        classes = ['normal', 'adenocarcinoma', 'large.cell', 'squamous']
        predicted_class = classes[np.argmax(prediction)]
        st.write('Prediction:', predicted_class)

        # Store user input data in session state
        st.session_state['user_input_data'] = {
            "Name": name,
            "Gender": gender,
            "Age": age,
            "Smoking Status": smoking_status
        }

        # Download the processed image
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        st.markdown(get_binary_file_downloader_html(buffer, 'Processed_Image', 'Download Processed Image'), unsafe_allow_html=True)

        if st.button("Provide Feedback"):
            st.session_state['page'] = 'feedback'

def feedback_page():
    st.title("Feedback")
    feedback = st.text_area("How was your experience?")
    
    # Display user input data table from model page
    st.write("You provided the following information:")
    user_input_data = st.session_state.get('user_input_data', {})
    user_input_df = pd.DataFrame([user_input_data])  # Convert dictionary to DataFrame
    st.table(user_input_df)

    if st.button("Submit Feedback"):
        # Here you could write the feedback to a file or database
        st.success("Thank you for your feedback!")
        st.session_state['page'] = 'login'

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

def get_binary_file_downloader_html(bin_data, file_label='File', button_text='Download'):
    bin_str = base64.b64encode(bin_data.getvalue()).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{file_label}.jpg">{button_text}</a>'
    return href

if __name__ == "__main__":
    main()

