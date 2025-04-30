# 🍎 Fruits and Vegetables Vision System

<div align="center">
  <img src="https://github.com/user-attachments/assets/061b5620-61b5-4724-a2e4-5ed6cf1139a8" alt="Project Logo" width="200"/>
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
  <img src="https://github.com/user-attachments/assets/4b610bff-6b1d-4e02-bf77-16eed4a46c57" alt="Demo" width="600"/>
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
| Fruit & Vegetable Classification | MobileNetV2 | Custom dataset with 30+ classes | 95% | [Open Notebook](https://colab.research.google.com/drive/14enajVCYpVDDurWDLTkJVu4DMAw3eHAO?usp=sharing) |
| Ripeness Detection | MobileNetV2 | Custom dataset with ripe/unripe examples | 92% | [Open Notebook](https://www.kaggle.com/code/geekerman/fruit-ripeness-best) |
| Freshness Analysis | MobileNetV2 | Custom dataset with fresh/rotten examples | 93% | [Open Notebook](https://www.kaggle.com/code/amoghansh21bec0608/fruit-rotten-mobilenet) |
| Item Counting | YOLOv8 | Custom annotated dataset | mAP 0.89 | [Open Notebook](https://www.kaggle.com/code/amoghansh21bec0608/fruits-final) |

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
python app.py
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
        <img src="https://github.com/user-attachments/assets/44e76007-a58c-4f22-823b-461c6beeef4a" alt="Example Results" width="600"/>
      </td>
    </tr>
  </table>
</div>

## 📁 Project Structure

```
FruitVegVision/
├── app.py                    # Main Flask application
├── models/                   # Trained model files
│   ├── model.h5              # Food classification model
│   ├── rotten.h5             # Freshness detection model
│   ├── counting.pt           # YOLOv8 counting model
│   ├── fruit_ripeness_model.h5  # Fruit ripeness model
│   └── vegetable_ripeness_model (1).h5  # Vegetable ripeness model
├── static/                   # Static files (CSS, JS, images)
│   └── uploads/              # Folder for uploaded images
├── templates/                # HTML templates
│   ├── index.html            # Main page
│   ├── results.html          # Results display page
│   └── error.html            # Error page
└── requirements.txt          # Project dependencies
```

## 🔮 Future Improvements

- [ ] Deploy the application to a cloud platform
- [ ] Add nutritional information for identified fruits and vegetables
- [ ] Implement real-time detection using a webcam
- [ ] Develop a mobile application version
- [ ] Add multi-language support

---

<div align="center">
  <p>Made with ❤️ by <a href="https://github.com/yourusername">Your Name</a></p>
</div>
