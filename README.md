![d3](https://github.com/user-attachments/assets/8c50dbaa-f1c9-4e9c-852c-518f4ae8cf79)# 🍎 Fruits and Vegetables Vision System

<div align="center">
  <img src="static/images/logo.png" alt="Project Logo" width="200"/>
  <br>
  <p><em>A comprehensive computer vision system for analyzing fruits and vegetables</em></p>
  
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://www.tensorflow.org/)
  [![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
  [![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-yellow.svg)](https://github.com/ultralytics/ultralytics)
  [![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
</div>

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Technologies Used](#-technologies-used)
- [Model Training](#-model-training)
- [Installation](#-installation)
- [Usage](#-usage)
- [Examples](#-examples)
- [Project Structure](#-project-structure)
- [Future Improvements](#-future-improvements)

## 🔍 Overview

This project uses deep learning to identify, analyze, and count fruits and vegetables in images. Four specialized models handle different tasks, creating a comprehensive vision system for produce analysis.

## ✨ Features

<table>
  <tr>
    <td width="25%">
      <h3 align="center">🍊 Classification</h3>
      <p align="center">Identifies 30+ types of fruits and vegetables with high accuracy</p>
    </td>
    <td width="25%">
      <h3 align="center">🟢 Ripeness</h3>
      <p align="center">Determines if produce is ripe or unripe</p>
    </td>
    <td width="25%">
      <h3 align="center">🥭 Freshness</h3>
      <p align="center">Detects whether produce is fresh or rotten</p>
    </td>
    <td width="25%">
      <h3 align="center">🔢 Counting</h3>
      <p align="center">Counts the number of fruits and vegetables in an image</p>
    </td>
  </tr>
</table>

## 🎬 Demo

<div align="center">
  ![d3](https://github.com/user-attachments/assets/4b610bff-6b1d-4e02-bf77-16eed4a46c57)
</div>

## 🔧 Technologies Used

- **Models**: 
  - MobileNetV2 for classification, ripeness, and freshness detection
  - YOLOv8 for object detection and counting
- **Backend**: Flask web application
- **Training**: Models trained on Google Colab

## 🧠 Model Training

All models were custom-trained on curated datasets:

| Model | Architecture | Dataset | Accuracy | Colab Notebook |
|-------|-------------|---------|----------|---------------|
| Fruit & Vegetable Classification | MobileNetV2 | Custom dataset with 30+ classes | 95% | [Open Notebook](https://colab.research.google.com/drive/your-notebook-https://colab.research.google.com/drive/14enajVCYpVDDurWDLTkJVu4DMAw3eHAO?usp=sharing) |
| Ripeness Detection | MobileNetV2 | Custom dataset with ripe/unripe examples | 92% | [Open Notebook](https://colab.research.google.com/drive/your-notebook-https://www.kaggle.com/code/geekerman/fruit-ripeness-best) |
| Freshness Analysis | MobileNetV2 | Custom dataset with fresh/rotten examples | 93% | [Open Notebook](https://colab.research.google.com/drive/your-notebook-https://www.kaggle.com/code/amoghansh21bec0608/fruit-rotten-mobilenet) |
| Item Counting | YOLOv8 | Custom annotated dataset | mAP 0.89 | [Open Notebook](https://colab.research.google.com/drive/your-notebook-https://www.kaggle.com/code/amoghansh21bec0608/fruits-final) |

## 📥 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/FruitVegVision.git
cd FruitVegVision

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download model files
# (Add instructions for downloading model files if they're not included in the repo)
```

## 🚀 Usage

Start the Flask application:

```bash
import os
import random
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from flask import Flask, render_template, request # type: ignore
from PIL import Image # type: ignore
import io
from werkzeug.utils import secure_filename # type: ignore
from ultralytics import YOLO # type: ignore

app = Flask(__name__, static_folder='static')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

FOOD_MODEL_PATH = os.path.join('models', 'model.h5')
ROTTEN_MODEL_PATH = os.path.join('models', 'rotten.h5')
COUNT_MODEL_PATH = os.path.join('models', 'counting.pt')
FRUIT_RIPENESS_MODEL_PATH = os.path.join('models', 'fruit_ripeness_model.h5')
VEG_RIPENESS_MODEL_PATH = os.path.join('models', 'vegetable_ripeness_model (1).h5')

food_model = None
rotten_model = None
counting_model = None
fruit_ripeness_model = None
veg_ripeness_model = None

food_classes = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
                'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger',
                'grapes', 'jalapeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika',
                'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'radish', 'soy beans', 'spinach',
                'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

rottenness_classes = ['fresh', 'rotten']
rotten_supported_foods = ['apple', 'banana', 'grape', 'guava', 'jujube', 'mango', 'orange',
                          'pomegranate', 'strawberry', 'tomato', 'bellpepper', 'carrot', 'cucumber', 'potato']
rotten_food_map = {
    'bell pepper': 'bellpepper',
    'grapes': 'grape'
}

fruits_supported_ripeness = ['banana', 'mango', 'apple', 'orange', 'papaya', 'pineapple', 'grape', 'guava', 'pomegranate', 'strawberry', 'dragonfruit']
vegetables_supported_ripeness = ['tomato', 'bell pepper', 'cucumber', 'chile pepper', 'new mexico green chile']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    global food_model, rotten_model, counting_model, fruit_ripeness_model, veg_ripeness_model
    try:
        required_files = [FOOD_MODEL_PATH, ROTTEN_MODEL_PATH, COUNT_MODEL_PATH,
                          FRUIT_RIPENESS_MODEL_PATH, VEG_RIPENESS_MODEL_PATH]
        if not all(os.path.exists(p) for p in required_files):
            raise FileNotFoundError("One or more models not found")

        food_model = tf.keras.models.load_model(FOOD_MODEL_PATH)
        rotten_model = tf.keras.models.load_model(ROTTEN_MODEL_PATH)
        counting_model = YOLO(COUNT_MODEL_PATH)
        counting_model.eval()
        fruit_ripeness_model = tf.keras.models.load_model(FRUIT_RIPENESS_MODEL_PATH)
        veg_ripeness_model = tf.keras.models.load_model(VEG_RIPENESS_MODEL_PATH)

        print(" All models loaded successfully.")
        return True
    except Exception as e:
        print(f" Model loading failed: {e}")
        return False

def preprocess_image(img, target_size=(224, 224)):
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    elif not isinstance(img, Image.Image):
        img = Image.open(io.BytesIO(img)).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def count_objects(img_path):
    try:
        results = counting_model.predict(source=img_path, conf=0.25, save=False)[0]
        class_counts = {}
        for cls_id in results.boxes.cls:
            label = counting_model.names[int(cls_id)]
            class_counts[label] = class_counts.get(label, 0) + 1
        return class_counts
    except Exception as e:
        print(f" Error in counting objects: {e}")
        return {}

def extract_ripeness_from_label(label):
    label = label.lower().replace("_", "")
    if "unripe" in label:
        return "Unripe"
    elif "ripe" in lasbel:
        return "Ripe"
    else:
        return "Unknown"

def predict_ripeness(img_array, food_name):
    try:
        label = food_name.lower()
        fruit_ripeness_classes = [
            'RipeApple', 'UnripeApple', 'RipeBanana', 'UnripeBanana',
            'RipeGrape', 'UnripeGrape', 'RipeGuava', 'UnripeGuava',
            'RipeMango', 'UnripeMango', 'RipeOrange', 'UnripeOrange',
            'RipePapaya', 'UnripePapaya', 'RipePineapple', 'UnripePineapple',
            'RipePomegranate', 'UnripePomegranate', 'RipeStrawberry', 'UnripeStrawberry',
            'RipeDragonFruit', 'UnripeDragonFruit'
        ]

        veg_ripeness_classes = [
            'Tomato_Ripe', 'Tomato_Unripe',
            'Bell Pepper_Ripe', 'Bell Pepper_Unripe',
            'Chile Pepper_Ripe', 'Chile Pepper_Unripe',
            'New Mexico Green Chile_Ripe', 'New Mexico Green Chile_Unripe'
        ]

        if label in fruits_supported_ripeness:
            model = fruit_ripeness_model
            classes = fruit_ripeness_classes
        elif label in vegetables_supported_ripeness:
            model = veg_ripeness_model
            classes = veg_ripeness_classes
        else:
            return "Unknown", 0.0

        prediction = model.predict(img_array, verbose=0)
        index = np.argmax(prediction[0])
        predicted_label = classes[index]
        ripeness = extract_ripeness_from_label(predicted_label)
        confidence = round(float(prediction[0][index]) * 100, 2)
        return ripeness, confidence

    except Exception as e:
        print(f"Ripeness prediction error: {e}")
        return "Error", 0.0

@app.route('/')
def index():
    model_status = {
        "food_model": "Loaded" if food_model else "Not loaded",
        "rotten_model": "Loaded" if rotten_model else "Not loaded",
        "counting_model": "Loaded" if counting_model else "Not loaded",
        "fruit_ripeness_model": "Loaded" if fruit_ripeness_model else "Not loaded",
        "veg_ripeness_model": "Loaded" if veg_ripeness_model else "Not loaded"
    }
    return render_template('index.html', model_status=model_status)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return render_template('error.html', error_message="No file part")

        file = request.files['file']
        if file.filename == '':
            return render_template('error.html', error_message="No selected file")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{random.randint(1000, 9999)}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)

            img_array = preprocess_image(filepath)

            if food_model is None or rotten_model is None or counting_model is None:
                return render_template('error.html', error_message="Models not loaded")

            food_prediction = food_model.predict(img_array, verbose=0)
            food_index = np.argmax(food_prediction[0])
            food_name = food_classes[food_index]
            food_display_name = food_name.capitalize()
            food_confidence = round(float(food_prediction[0][food_index]) * 100, 2)

            rottenness = "Healthy"
            rot_confidence = 85.0
            rotten_food_name = rotten_food_map.get(food_name, food_name)
            if rotten_food_name in rotten_supported_foods:
                rot_prediction = rotten_model.predict(img_array, verbose=0)
                rot_index = np.argmax(rot_prediction[0])
                if rot_index < len(rottenness_classes):
                    rot_class = rottenness_classes[rot_index]
                    rottenness = "Healthy" if rot_class == 'fresh' else "Rotten"
                    rot_confidence = round(float(rot_prediction[0][rot_index]) * 100, 2)

            ripeness, ripeness_confidence = predict_ripeness(img_array, food_name)
            object_count = count_objects(filepath)

            image_path = f"uploads/{unique_filename}"
            return render_template("results.html",
                                   image_path=image_path,
                                   food=food_display_name,
                                   food_confidence=food_confidence,
                                   rottenness=rottenness,
                                   rot_confidence=rot_confidence,
                                   ripeness=ripeness,
                                   ripeness_confidence=ripeness_confidence,
                                   object_count=object_count)
        else:
            return render_template('error.html', error_message="File type not allowed")

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return render_template('error.html', error_message=f"Unexpected error: {str(e)}")

if __name__ == '__main__':
    if load_models():
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Models not loaded. Exiting...")
        
```

Then open your browser and go to `http://localhost:5000`.

### Web Interface

1. Upload an image containing fruits and/or vegetables
2. Select the analysis you want to perform:
   - Identify fruits and vegetables
   - Check ripeness
   - Check freshness
   - Count items
3. View the results!

## 📸 Examples

<div align="center">
  <table>
    <tr>
      <td>
        <img src="static/images/examples/d7.png" alt="Classification Example" width="300"/>
      </td>
    </tr>
  </table>
</div>


## 🔮 Future Improvements

- [ ] Deploy the application to a cloud platform
- [ ] Add nutritional information for identified fruits and vegetables
- [ ] Implement real-time detection using a webcam
- [ ] Develop a mobile application version
- [ ] Add multi-language support
