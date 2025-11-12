# 🖐️ Indian Sign Language (ISL) Classification – Keypoint-Based System

## 🚀 Project Overview
This project implements a **real-time classification system for Indian Sign Language (ISL)** signs — supporting both **single-hand and two-hand gestures**.  
It transitions from an image-based **ResNet** approach to a **keypoint-based classification** pipeline using **MediaPipe** and a **Multi-Layer Perceptron (MLP)** for faster, more reliable results.

---

## 📊 Data Source
The dataset should be organized in the following format:

```
dataset/
├── train/
│   ├── A/
│   ├── B/
│   ├── C/
│   └── ...
└── val/
    ├── A/
    ├── B/
    ├── C/
    └── ...
```

📦 **Dataset:** [Indian Sign Language (ISL) Dataset – Kaggle](https://www.kaggle.com/datasets)

---

## ⚙️ Key Components

| File | Functionality |
|------|----------------|
| `data_preprocessor.py` | Converts images in `/dataset` into `keypoint_data.csv` containing extracted keypoints. |
| `model.py` | Defines the MLP (Multi-Layer Perceptron) architecture with 126 input features. |
| `train_mlp.py` | Trains the MLP model on the keypoint data using GPU acceleration. |
| `evaluate_model.py` | Evaluates the trained model on validation data. |
| `infer_realtime.py` | Runs real-time webcam-based inference and sign detection. |
| `utils.py` | Core logic for extracting 126 keypoints per hand using MediaPipe. |

---

## 💻 Setup and Installation

### 1. Prerequisites
Download and place the ISL dataset folders (A, B, C, G, etc.) into:

```
dataset/train/
dataset/val/
```

> 💡 **macOS Users:** Ensure the Apple-optimized TensorFlow stack is installed for GPU acceleration.

---

### 2. Environment Setup
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate
```

---

### 3. Install Dependencies
```bash
# Install packages from the requirements file
# Note: Ensure tensorflow-macos and tensorflow-metal are listed for Apple Silicon
pip install -r requirements.txt
```

---

## 🛠️ Execution Pipeline (Start to Finish)

### **Phase 1: Data Preparation (Image → Keypoints)**
Converts the image dataset into `keypoint_data.csv` for MLP training.

```bash
python data_preprocessor.py
# Output: keypoint_data.csv in the project root
```

---

### **Phase 2: Model Training (GPU Accelerated)**
Defines and trains the MLP model using GPU.

```bash
python train_mlp.py
# Output:
# models/sign_model_mlp_saved/
# models/classes.txt
```

---

### **Phase 3: Model Evaluation**
Evaluates trained model on validation data.

```bash
python evaluate_model.py
# Output: Validation Accuracy and Classification Report
```

---

### **Phase 4: Real-Time Inference**
Run real-time detection through your webcam.

```bash
# Basic real-time inference
python infer_realtime.py

# With Text-to-Speech (TTS) output
python infer_realtime.py --use_tts
```

---

## 🧱 Project Structure
```
SIGN_LANGUAGE_PROJECT/
├── dataset/
│   ├── train/
│   └── val/
├── models/
│   └── sign_model_mlp_saved/
├── keypoint_data.csv
├── data_preprocessor.py
├── train_mlp.py
├── evaluate_model.py
├── infer_realtime.py
├── model.py
├── utils.py
└── requirements.txt
```

---

