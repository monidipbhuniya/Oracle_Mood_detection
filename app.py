from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
import cv2

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('facial_emotion_detection_model.h5')

# Define class names (order must match training)
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Preprocess base64 image
def preprocess_image(img_b64):
    try:
        # Decode base64 string
        img_data = base64.b64decode(img_b64)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)  # grayscale

        # Resize to 48x48
        img_resized = cv2.resize(img, (48, 48))

        # Normalize
        img_normalized = img_resized / 255.0

        # Expand dimensions for model (1, 48, 48, 1)
        img_array = np.expand_dims(img_normalized, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)

        return img_array
    except Exception as e:
        raise ValueError(f"Image preprocessing error: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        img_b64 = data.get("image", None)

        if not img_b64:
            return jsonify({"error": "No image provided"}), 400

        # Preprocess image
        img_array = preprocess_image(img_b64)

        # Predict
        prediction = model.predict(img_array)
        predicted_index = int(np.argmax(prediction))
        predicted_class = class_names[predicted_index]
        confidence = round(float(prediction[0][predicted_index]) * 100, 2)

        return jsonify({
            "emotion": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # For development; use gunicorn in production
    app.run(host="0.0.0.0", port=5000)
