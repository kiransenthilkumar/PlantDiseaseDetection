from flask import Flask, render_template, request, redirect, Response, send_file
from fpdf import FPDF
import tensorflow as tf
import numpy as np
import cv2
import csv
import json
import os
import sqlite3
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_PATH = "fruit_disease_model.keras"
CLASS_INDEX_PATH = "class_indices.json"

IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 60
PER_PAGE = 10

# ---------------- Load model ----------------
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)

class_names = {v: k for k, v in class_indices.items()}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- Treatments ----------------
TREATMENTS = {
    "Apple___Apple_scab": "Apply captan or myclobutanil fungicides.",
    "Apple___Black_rot": "Remove infected fruits and prune branches.",
    "Apple___Cedar_apple_rust": "Remove nearby juniper plants.",
    "Apple___healthy": "Healthy plant.",

    "Blueberry___healthy": "Healthy plant.",

    "Cherry_(including_sour)___Powdery_mildew": "Apply sulfur fungicide.",
    "Cherry_(including_sour)___healthy": "Healthy plant.",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Use resistant hybrids.",
    "Corn_(maize)___Common_rust_": "Apply fungicides if severe.",
    "Corn_(maize)___Northern_Leaf_Blight": "Practice crop rotation.",
    "Corn_(maize)___healthy": "Healthy crop.",

    "Grape___Black_rot": "Remove infected berries.",
    "Grape___Esca_(Black_Measles)": "Prune infected vines.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply fungicides.",
    "Grape___healthy": "Healthy vine.",

    "Orange___Haunglongbing_(Citrus_greening)": "Remove infected trees.",

    "Peach___Bacterial_spot": "Apply copper sprays.",
    "Peach___healthy": "Healthy plant.",

    "Pepper,_bell___Bacterial_spot": "Use disease-free seeds.",
    "Pepper,_bell___healthy": "Healthy plant.",

    "Potato___Early_blight": "Apply chlorothalonil fungicide.",
    "Potato___Late_blight": "Remove infected plants.",
    "Potato___healthy": "Healthy crop.",

    "Raspberry___healthy": "Healthy plant.",
    "Soybean___healthy": "Healthy plant.",
    "Squash___Powdery_mildew": "Apply neem oil or sulfur.",
    "Strawberry___Leaf_scorch": "Remove infected leaves.",
    "Strawberry___healthy": "Healthy plant.",

    "Tomato___Bacterial_spot": "Apply copper fungicide.",
    "Tomato___Early_blight": "Remove infected leaves.",
    "Tomato___Late_blight": "Destroy infected plants.",
    "Tomato___Leaf_Mold": "Improve ventilation.",
    "Tomato___Septoria_leaf_spot": "Apply fungicide.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Use neem oil.",
    "Tomato___Target_Spot": "Avoid overhead watering.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies.",
    "Tomato___healthy": "Healthy plant."
}

# ---------------- Database ----------------
def init_db():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image TEXT,
            disease TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------- Routes ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = confidence = image_path = treatment = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            filename = f"{uuid.uuid4()}_{file.filename}"
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(image_path)

            img = cv2.imread(image_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

            preds = model.predict(img)
            class_index = np.argmax(preds)
            confidence = float(np.max(preds) * 100)

            if confidence < CONFIDENCE_THRESHOLD:
                prediction = "Uncertain prediction. Please upload a clearer image."
            else:
                prediction = class_names[class_index]
                treatment = TREATMENTS.get(prediction)

                conn = sqlite3.connect("predictions.db")
                c = conn.cursor()
                c.execute(
                    "INSERT INTO history (image, disease, confidence) VALUES (?, ?, ?)",
                    (filename, prediction, confidence)
                )
                conn.commit()
                conn.close()

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           treatment=treatment,
                           image_path=image_path)


@app.route("/history")
def history():
    page = int(request.args.get("page", 1))
    offset = (page - 1) * PER_PAGE

    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM history")
    total_records = c.fetchone()[0]

    c.execute("""
        SELECT id, image, disease, confidence, timestamp
        FROM history
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
    """, (PER_PAGE, offset))

    records = c.fetchall()
    conn.close()

    total_pages = (total_records + PER_PAGE - 1) // PER_PAGE

    return render_template("history.html",
                           records=records,
                           page=page,
                           total_pages=total_pages)


@app.route("/delete/<int:record_id>", methods=["POST"])
def delete_record(record_id):
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("DELETE FROM history WHERE id=?", (record_id,))
    conn.commit()
    conn.close()
    return redirect("/history")


@app.route("/clear-history", methods=["POST"])
def clear_history():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("DELETE FROM history")
    conn.commit()
    conn.close()
    return redirect("/history")


@app.route("/export/csv")
def export_csv():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("SELECT image, disease, confidence, timestamp FROM history")
    rows = c.fetchall()
    conn.close()

    def generate():
        yield "Image,Disease,Confidence,Timestamp\n"
        for r in rows:
            yield f"{r[0]},{r[1]},{round(r[2],2)},{r[3]}\n"

    return Response(generate(),
                    mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=history.csv"})


@app.route("/export/pdf")
def export_pdf():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("SELECT image, disease, confidence, timestamp FROM history")
    rows = c.fetchall()
    conn.close()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, "Plant Disease Prediction History", ln=True)

    for r in rows:
        pdf.multi_cell(0, 8,
            f"Image: {r[0]}\nDisease: {r[1]}\nConfidence: {round(r[2],2)}%\nDate: {r[3]}\n-----"
        )

    pdf_path = "prediction_history.pdf"
    pdf.output(pdf_path)
    return send_file(pdf_path, as_attachment=True)


@app.route("/analytics")
def analytics():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("""
        SELECT disease, COUNT(*)
        FROM history
        GROUP BY disease
        ORDER BY COUNT(*) DESC
    """)
    data = c.fetchall()
    conn.close()

    labels = [d[0] for d in data]
    values = [d[1] for d in data]

    return render_template("analytics.html", labels=labels, values=values)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

