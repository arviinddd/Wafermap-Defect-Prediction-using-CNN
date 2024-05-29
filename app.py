from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Load the pre-trained Keras model
model = load_model('wafermap.keras')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Decode the base64 image data to an image
    images_data = data['images']  # Assume this is a list of base64 encoded strings
    images = np.array([decode_image(base64_str) for base64_str in images_data])

    # Extract numerical data from JSON
    numerical_data = np.array([data['dieSize'], data['waferMapDim_x'], data['waferMapDim_y']])
    
    # Reshape numerical data if necessary
    numerical_data = numerical_data.reshape(1, -1)  # Assuming a single prediction

    # Make prediction using the preprocessed image data and numerical data
    predictions = model.predict([images, numerical_data])
    return jsonify(predictions.tolist())

def decode_image(base64_str):
    # Decode the base64 string
    img_data = base64.b64decode(base64_str)
    # Convert binary data to image
    img = Image.open(BytesIO(img_data))
    # Convert image to grayscale
    img = img.convert('L')
    # Resize the image as required by your model
    img = img.resize((32, 32))
    img_array = np.array(img)
    # Ensure the image has the shape (32, 32, 1)
    img_array = img_array.reshape(32, 32, 1)
    return img_array

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
