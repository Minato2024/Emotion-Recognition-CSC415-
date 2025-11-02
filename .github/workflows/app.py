# app.py
from fastapi import FastAPI, Form, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import sqlite3
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = FastAPI()

# --- File Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_emotionModel.h5")
DB_PATH = os.path.join(BASE_DIR, "database.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Load Model ---
model = load_model(MODEL_PATH)

# Emotion labels (7 classes)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# --- Initialize Database ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            image_path TEXT NOT NULL,
            emotion TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- Serve Static Files & Templates ---
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Helper Function: Save user data ---
def save_to_db(name, department, image_path, emotion):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (name, department, image_path, emotion) VALUES (?, ?, ?, ?)",
        (name, department, image_path, emotion)
    )
    conn.commit()
    conn.close()

# --- Helper Function: Predict Emotion ---
def predict_emotion(img_path):
    img = Image.open(img_path).convert('L')  # grayscale
    img = img.resize((48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    emotion_index = np.argmax(prediction)
    emotion = EMOTIONS[emotion_index]

    # Generate response message
    emotion_messages = {
        'Happy': "You look happy! Keep smiling 😊",
        'Sad': "You seem sad. Everything okay?",
        'Angry': "You look angry. Take a deep breath 😤",
        'Fear': "You seem scared. Don't worry, you're safe!",
        'Disgust': "Hmm... something bothering you?",
        'Surprise': "Whoa! You look surprised 😮",
        'Neutral': "You seem calm and composed 😌"
    }
    return emotion_messages.get(emotion, "Emotion detected."), emotion

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    name: str = Form(...),
    department: str = Form(...),
    file: UploadFile = File(...)
):
    # Save uploaded image
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Predict emotion
    result_message, emotion = predict_emotion(file_location)

    # Save to database
    image_url = f"/static/uploads/{file.filename}"
    save_to_db(name, department, image_url, emotion)

    # Return result to template
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": f"{result_message}",
            "image_url": image_url
        }
    )