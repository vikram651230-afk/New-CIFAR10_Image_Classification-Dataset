# CIFAR-10 Image Classification — Flask API Web Application

A complete, production-grade image classification web application and REST API built with **Flask** and **TensorFlow/Keras**, strictly implementing the non-convolutional **Artificial Neural Network (ANN)** architecture developed in `notebooks/CIFAR 10 image classification.ipynb`.

---

## 📌 Project Overview

This project provides an end-to-end deep learning web service for classifying 32×32 pixel images across the 10 standard CIFAR-10 categories. 

### 🛡️ Non-CNN Architecture Compliance
As specified in the project requirements, **zero 2D convolutions (Conv2D)** and **zero pooling layers** are used anywhere in the model. The model is a pure Deep Artificial Neural Network (Multilayer Perceptron):
- **Input:** $32 \times 32 \times 3$ RGB images (3,072 dimensions)
- **Flatten Layer:** Reshapes $(B, 32, 32, 3)$ to $(B, 3072)$
- **Hidden Layer 1:** Dense 200 units, ReLU activation
- **Hidden Layer 2:** Dense 100 units, ReLU activation
- **Hidden Layer 3:** Dense 30 units, ReLU activation
- **Output Layer:** Dense 10 units, Softmax activation

---

## 🗂️ CIFAR-10 Dataset Information

The **CIFAR-10** (Canadian Institute For Advanced Research) dataset consists of 60,000 $32 \times 32$ colour images in 10 mutually exclusive classes:

| Class Index | Class Label | Emoji | Description |
| :---: | :--- | :---: | :--- |
| **0** | `airplane` | ✈️ | Airplanes, passenger jets |
| **1** | `automobile` | 🚗 | Sedans, passenger cars |
| **2** | `bird` | 🐦 | Birds of various species |
| **3** | `cat` | 🐱 | Felines |
| **4** | `deer` | 🦌 | Woodland deer |
| **5** | `dog` | 🐶 | Canines |
| **6** | `frog` | 🐸 | Amphibians |
| **7** | `horse` | 🐴 | Equines |
| **8** | `ship` | 🚢 | Boats, cargo ships |
| **9** | `truck` | 🚛 | Commercial trucks |

- **Training set:** 50,000 images
- **Test set:** 10,000 images (held out for objective evaluation)
- **Preprocessing:** Pixel normalization ($x / 255.0$), RGB conversion, and resampling to $(32, 32)$

---

## 🏛️ Project Directory Structure

```text
cifar10-image-classification/
├── app/
│   ├── __init__.py               # Flask application factory with error handlers
│   ├── config.py                 # Central configuration and class definitions
│   ├── routes.py                 # Web interface routes and REST API endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── model_service.py      # Singleton thread-safe model loader
│   │   ├── image_service.py      # Standardized image preprocessing & Base64 preview
│   │   └── prediction_service.py # Inference pipeline & probability extraction
│   ├── utils/
│   │   ├── __init__.py
│   │   └── validators.py         # File extension and upload security validators
│   ├── templates/
│   │   └── index.html            # Responsive UI (drag-and-drop, preview, results)
│   └── static/
│       ├── css/
│       │   └── style.css         # Modern dark neumorphic stylesheet
│       ├── js/
│       │   └── script.js         # Asynchronous drag-and-drop & API fetch client
│       └── samples/              # 10 real CIFAR-10 sample test images
├── models/
│   ├── image_classification.keras # Trained model artifact (Keras v3 format)
│   ├── image_classification.h5    # Trained model artifact (HDF5 format)
│   ├── metrics.json              # Evaluated test accuracy and per-class stats
│   ├── confusion_matrix.png      # 10x10 test confusion matrix plot
│   ├── training_curves.png       # Loss and accuracy learning curves
│   └── train_model.py            # CLI script to train model from scratch
├── notebooks/
│   └── CIFAR 10 image classification.ipynb # Original research notebook
├── tests/
│   ├── conftest.py               # Pytest client fixtures and test configuration
│   ├── test_api.py               # REST API endpoint unit & integration tests
│   └── test_prediction.py        # Model loading, parity, & inference tests
├── uploads/                      # Temporary storage for user uploads
├── .env.example                  # Environment variable template
├── .gitignore                    # Git ignore file
├── pytest.ini                    # Pytest configuration
├── requirements.txt              # Pinned Python package dependencies
├── run.py                        # Web application entry point
└── README.md                     # Comprehensive documentation
```

---

## 🛠️ Technology Stack

- **Backend Framework:** Python 3.11+, Flask 3.0+
- **Deep Learning:** TensorFlow 2.15+, Keras 3.0+
- **Image Processing:** Pillow (PIL)
- **Numerical Computing & Metrics:** NumPy, Scikit-learn
- **Visualization:** Matplotlib
- **Testing:** Pytest
- **Frontend:** HTML5, CSS3, JavaScript (ES6+), Bootstrap 5.3

---

## 🚀 Installation & Setup Instructions

### 1. Clone or Open the Repository
```powershell
cd E:\CIFAR10_Image_Classification
```

### 2. Create and Activate Virtual Environment (Recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## ⚡ How to Run the Application

### Start the Flask Server
```powershell
python run.py
```
By default, the server will start at:
👉 **`http://127.0.0.1:5000`**

Open your web browser and navigate to `http://127.0.0.1:5000` to interact with the visual interface.

---

## 🧪 Retraining the Model (Optional)

To retrain the model from scratch using the exact notebook configuration:
```powershell
# Standard 10-epoch training (~1 minute)
python models/train_model.py --epochs 10 --batch-size 32

# Or train with fewer/more epochs
python models/train_model.py --epochs 5
```
This automatically updates `models/image_classification.keras`, `models/metrics.json`, and `models/confusion_matrix.png`.

---

## 📡 REST API Documentation

### 1. Health Check
- **Endpoint:** `GET /api/health`
- **Description:** Returns the operational status of the service and model loading state.
- **Sample Response:**
```json
{
  "convolutional_layers": 0,
  "model_architecture": "Pure-ANN (Flatten + 3 Dense Relu + Dense Softmax)",
  "model_loaded": true,
  "status": "healthy",
  "supported_classes": [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
  ],
  "version": "1.0.0"
}
```

---

### 2. Supported Classes
- **Endpoint:** `GET /api/classes`
- **Description:** Returns the 10 CIFAR-10 class labels with indices and emojis.
- **Sample Response:**
```json
{
  "total_classes": 10,
  "classes": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
  "detailed": [
    {"index": 0, "name": "airplane", "emoji": "✈️"},
    {"index": 1, "name": "automobile", "emoji": "🚗"},
    {"index": 2, "name": "bird", "emoji": "🐦"}
  ]
}
```

---

### 3. Image Classification
- **Endpoint:** `POST /api/predict`
- **Content-Type:** `multipart/form-data`
- **Parameters:** `image` (binary file: PNG, JPG, JPEG, WEBP, BMP)
- **Sample cURL Request:**
```bash
curl -X POST -F "image=@sample_image.png" http://127.0.0.1:5000/api/predict
```
- **Sample JSON Response:**
```json
{
  "success": true,
  "filename": "sample_image.png",
  "predicted_class": "frog",
  "predicted_index": 6,
  "emoji": "🐸",
  "confidence_percentage": 58.42,
  "confidence_score": 0.5842,
  "latency_ms": 28.14,
  "model_architecture": "Pure-ANN (No Convolutions)",
  "input_dimensions": "32x32x3",
  "top_3": [
    {"class": "frog", "emoji": "🐸", "percentage": 58.42, "probability": 0.5842},
    {"class": "cat", "emoji": "🐱", "percentage": 14.21, "probability": 0.1421},
    {"class": "deer", "emoji": "🦌", "percentage": 9.15, "probability": 0.0915}
  ],
  "class_probabilities": {
    "airplane": 0.0214,
    "automobile": 0.0105,
    "bird": 0.0612,
    "cat": 0.1421,
    "deer": 0.0915,
    "dog": 0.0481,
    "frog": 0.5842,
    "horse": 0.0125,
    "ship": 0.0082,
    "truck": 0.0203
  }
}
```

---

### 4. Model Evaluation Metrics
- **Endpoint:** `GET /api/metrics`
- **Description:** Returns the evaluated performance on the 10,000 test set images.

---

## 🚦 Automated Testing Instructions

Execute the comprehensive test suite covering API routes, security validators, model parity, and prediction pipelines:
```powershell
pytest tests/ -v
```

All 15 tests verify:
- Flask initialization and configuration
- Model loading as thread-safe singleton
- Architectural compliance (0 convolution or pooling layers)
- Image preprocessing to $(1, 32, 32, 3)$
- File validation (empty files, invalid extensions)
- Valid prediction output formatting and class probability sums
"# New-CIFAR10_Image_Classification-Dataset" 
