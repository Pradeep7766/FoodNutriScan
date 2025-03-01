import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from PIL import Image
import io

# Load nutrition data
nutrition_data = pd.read_csv("nutrition_info.csv")

# Tensorflow Model Prediction
def model_prediction(test_image, confidence_threshold=90):
    model = tf.keras.models.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    confidence_scores = predictions[0]
    max_confidence = np.max(confidence_scores)
    predicted_class_index = np.argmax(confidence_scores)
    
    confidence_threshold_decimal = confidence_threshold / 100.0
    
    if max_confidence >= confidence_threshold_decimal:
        return predicted_class_index, max_confidence
    else:
        return None, max_confidence

st.set_page_config(layout="centered")

# Custom CSS
st.markdown(
    """
    <style>
    body {
        background-image: url("web.jpg"); 
        background-size: cover;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Prediction"])

# Main Page
if app_mode == "Home":
    st.header("FOODNUTRISCAN")
    st.header("FOOD & FRUITS RECOGNITION SYSTEM")
    image_path = r"web.jpg"

    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.warning("Image not found. Please check the path.")

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    
    camera_image = st.camera_input("Take a picture with your camera:")
    uploaded_image = st.file_uploader("or Choose an Image:", type=["jpg", "jpeg", "png"])
    
    test_image = None
    if uploaded_image:
        test_image = uploaded_image
    elif camera_image:
        img_bytes = io.BytesIO(camera_image.read())
        image = Image.open(img_bytes)
        image.save("temp.jpg")
        test_image = "temp.jpg"

    if test_image:
        if isinstance(test_image, str):  # File path
            image = Image.open(test_image)
        else:  # Uploaded file
            image = Image.open(test_image)
        
        st.image(image, use_container_width=True)

        if st.button("Predict"):
            with st.spinner("Making a prediction..."):
                result_index, confidence = model_prediction(test_image)

                if result_index is not None:
                    with open("labels.txt") as f:
                        content = f.readlines()
                    label = [i.strip() for i in content]
                    food_item = label[result_index]
                    
                    st.success(f"Model predicts it's a **{food_item}** with confidence **{confidence * 100:.2f}%**")

                    nutrition_info = nutrition_data[nutrition_data['Food'] == food_item]
                    if not nutrition_info.empty:
                        st.subheader("Nutrition Information:")
                        st.table(nutrition_info)
                    else:
                        st.warning("Nutrition information not found for this item.")
                else:
                    st.warning(f"Model couldn't confidently predict the item. Confidence was **{confidence * 100:.2f}%**.")
