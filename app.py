import logging
from flask import Flask, render_template, request, jsonify
import base64
import io
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from model import DeepCNN  # Import your PyTorch model here
from config import MODEL_PATH
from werkzeug.exceptions import BadRequest
from flask import current_app

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the pre-trained PyTorch model
def get_model():
    model = DeepCNN()
    model.load_state_dict(torch.load('mnist_model.pth', map_location=torch.device('cpu')))  # Load the model
    model.eval()  # Set the model to evaluation mode
    return model

model = get_model()  # Load the model once

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            image_data = data['image_data'].split(',')[1]  # Removing "data:image/png;base64," from the data URL
            img = Image.open(io.BytesIO(base64.b64decode(image_data)))
            img = img.convert('L')  # Convert to grayscale
            img = img.resize((28, 28))  # Resize image to match the model input size

            # Preprocess the image for prediction using PyTorch transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            img = transform(img).unsqueeze(0)

            # Make prediction using the loaded PyTorch model
            with torch.no_grad():
                outputs = model(img)
                _, predicted_digit = torch.max(outputs.data, 1)

            return jsonify({'predicted_digit': int(predicted_digit)})
        except BadRequest:
            current_app.logger.error("Bad request", exc_info=True)
            return jsonify({'error': 'Bad request, please check the data sent.'}), 400
        except Exception as e:
            current_app.logger.error("Prediction failed", exc_info=True)
            return jsonify({'error': 'Prediction failed, please try again.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
