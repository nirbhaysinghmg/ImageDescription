from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the pre-trained image classification model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

def prepare_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((224, 224))
    image = np.array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

def get_image_description(image_bytes):
    image = prepare_image(image_bytes)
    predictions = model.predict(image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
    description = decoded_predictions[0][1]
    return description

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Process the image
    image_bytes = file.read()
    description = get_image_description(image_bytes)
    return jsonify({"description": description})

if __name__ == '__main__':
    app.run(debug=True)
