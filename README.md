# 🌿 Plant Disease Detection System using Deep Learning

## 📌 Project Overview
Plant diseases significantly reduce agricultural productivity and farmer income.  
This project presents a **web-based Plant Disease Detection System** that uses **Deep Learning (CNN + MobileNet-ready architecture)** to automatically identify plant leaf diseases from images and suggest appropriate treatments.

The system is designed to be:
- Accurate
- User-friendly
- Mobile-compatible
- Suitable for real-world agricultural assistance

---

## 🎯 Objectives
- Detect plant leaf diseases using image classification
- Provide confidence-based predictions
- Suggest disease-specific treatments
- Maintain prediction history
- Visualize disease analytics
- Deploy the system on cloud platforms

---

## 🧠 Technologies Used

### Software
- **Python 3.10**
- **TensorFlow / Keras**
- **Flask (Web Framework)**
- **OpenCV**
- **SQLite (History Storage)**
- **Chart.js (Analytics)**
- **Gunicorn (Production Server)**

### Hardware
- Minimum: Dual-core CPU, 8GB RAM
- Recommended: Multi-core CPU / GPU (for training)
- Camera / Mobile device for leaf image capture

---

## 🗂 Dataset
- **PlantVillage Dataset**
- 38 plant disease classes
- Dataset is **NOT uploaded** to GitHub (ignored via `.gitignore`)

---

## 🏗 System Architecture
1. User uploads leaf image
2. Image preprocessing (resize, normalize)
3. CNN-based prediction model
4. Confidence threshold validation
5. Disease identification
6. Treatment suggestion
7. History stored in SQLite
8. Analytics generated from history

---

## 🔧 Key Features
- 🌱 Multi-crop disease detection
- 📊 Confidence threshold warning
- 💊 Disease-wise treatment suggestions
- 📜 Prediction history with images
- 📈 Disease analytics dashboard
- 📤 Export history (CSV & PDF)
- 📱 Mobile-friendly UI

---

## 📂 Project Structure
PlantDiseaseDetection/
│
├── app.py
├── train.py
├── predict.py
├── split_dataset.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│ ├── index.html
│ ├── history.html
│ ├── analytics.html
│
├── static/
│ └── uploads/
│
└── dataset/ (ignored)



---

## ▶ How to Run Locally

### 1️⃣ Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

pip install -r requirements.txt
python app.py
http://127.0.0.1:5000



---

# 🚀 Deployment Guide (Gunicorn + Render)

---

## 🔧 Step 1: Update `requirements.txt`

Make sure **Gunicorn** is included:

```txt
flask
tensorflow==2.15.0
keras==2.15.0
numpy
opencv-python
pillow
matplotlib
fpdf
gunicorn

