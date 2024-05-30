import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

model = load_model('wafermap.keras')

def decode_image(image_file):
    img = Image.open(image_file)
    img = img.convert('L')  
    img = img.resize((32, 32))  
    img_array = np.array(img)
    img_array = img_array.reshape(32, 32, 1)  
    return img_array


st.title('Wafer Image Prediction')


uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    decoded_image = decode_image(uploaded_file)


die_size = st.number_input("Enter Die Size:", min_value=0.0, value=1.0, step=0.1)
wafer_map_dim_x = st.number_input("Enter Wafer Map Dimension X:", min_value=1, value=32)
wafer_map_dim_y = st.number_input("Enter Wafer Map Dimension Y:", min_value=1, value=32)


failure_types = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']


if st.button('Predict'):
    if uploaded_file is not None:
        
        numerical_data = np.array([die_size, wafer_map_dim_x, wafer_map_dim_y])
        numerical_data = numerical_data.reshape(1, -1)
        
        
        predictions = model.predict([np.array([decoded_image]), numerical_data])
        predicted_index = np.argmax(predictions)
        predicted_failure_type = failure_types[predicted_index]
        
        st.write("Predicted Failure Type:", predicted_failure_type)
    else:
        st.error("Please upload an image for prediction.")
