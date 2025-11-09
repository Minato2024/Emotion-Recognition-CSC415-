# Emotion Detection Web App

A clean and simple Flask web application for detecting human emotions from uploaded images using artificial intelligence.

## 🎭 Features

- **Image Upload**: Simple drag & drop or file upload interface
- **AI-Powered Detection**: Uses Hugging Face's `dima806/facial_emotions_image_detection` model
- **Real-time Results**: Get instant emotion predictions with confidence scores
- **Clean Interface**: Simple, user-friendly Flask web interface
- **Statistics Dashboard**: View prediction history and basic analytics
- **SQLite Database**: Automatic logging of all predictions
- **RESTful API**: JSON endpoints for integration

## 😊 Supported Emotions

- **Happy** 😊
- **Sad** 😢
- **Angry** 😠
- **Surprise** 😲
- **Fear** 😨
- **Disgust** 🤢
- **Neutral** 😐

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Internet connection (for first-time model download)

### Installation

1. **Navigate to the project directory**
   ```bash
   cd SOBOYEJO-OLUWALASE_23CD034363_EMOTION_DETECTION_WEB_APP
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python run.py
   ```

4. **Open your browser**
   - Go to: http://localhost:5000 (or the port shown in terminal)
   - Upload an image and get instant emotion detection!

### Alternative Run Methods

```bash
# Direct Flask run
python app.py

# Or with Flask command
flask run
```

## 📁 Project Structure

```
SOBOYEJO-OLUWALASE_23CD034363_EMOTION_DETECTION_WEB_APP/
├── app.py              # Main Flask application
├── model.py            # Emotion detection model handler
├── run.py              # Smart launcher script
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── templates/         # HTML templates
│   └── index.html     # Main web interface
├── static/            # CSS styles
│   └── style.css      # Additional styling
├── uploads/           # Temporary upload folder
└── emotion_predictions.db  # SQLite database (auto-created)
```

## 🔧 API Endpoints

- **GET /** - Main web interface
- **POST /predict** - Upload image for emotion prediction
- **GET /health** - Health check and system status
- **GET /stats** - Prediction statistics (JSON)

## 📊 Usage Example

### Web Interface
1. Open http://localhost:5000 in your browser
2. Click "Choose File" or drag & drop an image
3. Wait for analysis (usually 1-3 seconds)
4. View the detected emotion and confidence score
5. Check the dashboard for prediction history

### API Usage
```bash
# Health check
curl http://localhost:5000/health

# Predict emotion from image
curl -X POST -F "file=@path/to/image.jpg" http://localhost:5000/predict

# Get statistics
curl http://localhost:5000/stats
```

### Example API Response
```json
{
    "success": true,
    "predicted_emotion": "happy",
    "confidence": 0.892,
    "all_scores": {
        "angry": 0.012,
        "disgust": 0.005,
        "fear": 0.018,
        "happy": 0.892,
        "neutral": 0.045,
        "sad": 0.015,
        "surprise": 0.013
    },
    "filename": "sample_image.jpg",
    "timestamp": "2025-01-15 14:30:25"
}
```

## 🛠️ Technical Details

- **Framework**: Flask (Python web framework)
- **AI Model**: HuggingFace Transformers (`dima806/facial_emotions_image_detection`)
- **Database**: SQLite (for storing prediction history)
- **Frontend**: HTML, CSS, JavaScript (Bootstrap for styling)
- **Image Processing**: PIL (Python Imaging Library)

## 📋 Requirements

The application requires the following Python packages:

- **Flask** (web framework)
- **PyTorch** (machine learning framework)
- **Transformers** (Hugging Face models)
- **Pillow** (image processing)
- **NumPy** (numerical computing)

See `requirements.txt` for specific versions.

## 🔍 Model Information

- **Model**: `dima806/facial_emotions_image_detection`
- **Type**: Image classification model trained on facial expressions
- **Input**: RGB images (automatically resized and preprocessed)
- **Output**: 7 emotion classes with confidence scores
- **Size**: ~500MB (downloads automatically on first run)

## 📈 Database Schema

The app automatically creates an SQLite database with the following structure:

```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    predicted_emotion TEXT NOT NULL,
    confidence REAL NOT NULL,
    image_path TEXT,
    source TEXT DEFAULT 'unknown'
);
```

## 🚨 Troubleshooting

### Common Issues

1. **Model Download Fails**
   - Ensure stable internet connection
   - The model (500MB+) downloads on first run
   - Check available disk space

2. **"Module not found" Error**
   ```bash
   pip install -r requirements.txt
   ```

3. **Port Already in Use**
   - The launcher automatically finds available ports
   - Default tries: 5000, 5001, 5002...
   - Or manually specify: `python app.py` and edit the port in code

4. **Image Upload Fails**
   - Check file format (supported: PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP)
   - Maximum file size: 16MB
   - Ensure image contains a visible face for best results

### System Requirements

- **RAM**: Minimum 4GB (8GB recommended for better performance)
- **Disk Space**: ~2GB (for model and dependencies)
- **CPU**: Any modern CPU (GPU optional but not required)

## 🎯 Best Results Tips

1. **Image Quality**: Use clear, well-lit images
2. **Face Visibility**: Ensure the face is clearly visible and not obstructed
3. **Image Size**: Larger images work better (automatically resized internally)
4. **Single Face**: Works best with one clear face in the image

## 🎨 Supported File Formats

- **Images**: PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP
- **Max Size**: 16MB per file
- **Processing**: Automatic RGB conversion and resizing

## 📝 License

This project is for educational and demonstration purposes.

## 🤝 Contributing

This is a simplified educational project. Feel free to fork and modify for your own use.

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all requirements are installed correctly
3. Verify your Python version (3.8+)

---

## 🚀 Quick Commands Summary

```bash
# Install and run
pip install -r requirements.txt
python run.py

# Direct Flask run
python app.py

# Check if everything works
python model.py
```

**Enjoy detecting emotions with AI!** 🎭✨