import os
import sys
import shutil
import numpy as np
import librosa
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ==========================================
# --- 1. CONFIGURATION ---
# ==========================================
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODEL_PATH = r"e:\DeepFake_Voice\asv5_detector.h5"
SR = 16000
DURATION = 4
N_MELS = 128
MAX_TIME_STEPS = 128

# Create the FastAPI App
app = FastAPI(title="Deepfake Voice Detector API")

# Enable CORS (Important so your frontend can communicate with this backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# --- 2. LOAD MODEL ON STARTUP ---
# ==========================================
print("🧠 System Boot: Loading AI Model into Memory...")
if not os.path.exists(MODEL_PATH):
    print(f"❌ Critical Error: Model not found at {MODEL_PATH}")
    sys.exit(1)
    
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ AI Model successfully loaded and ready for requests!")
print("🌐 Backend server is running at: http://127.0.0.1:8000")
print("🧪 You can test the API visually at: http://127.0.0.1:8000/docs")

# ==========================================
# --- 3. PREPROCESSING FUNCTION ---
# ==========================================
def load_and_preprocess(file_path):
    try:
        audio, _ = librosa.load(file_path, sr=SR, duration=DURATION)
    except Exception as e:
        raise ValueError(f"Could not read audio file: {str(e)}")

    if len(audio) < SR * DURATION:
        padding = SR * DURATION - len(audio)
        audio = np.pad(audio, (0, padding), 'constant')

    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    if mel_spectrogram.shape[1] < MAX_TIME_STEPS:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, MAX_TIME_STEPS - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :MAX_TIME_STEPS]

    # Return shape (1, 128, 128, 1)
    return np.expand_dims(np.expand_dims(mel_spectrogram, axis=-1), axis=0)

# ==========================================
# --- 4. API ENDPOINTS ---
# ==========================================
@app.get("/")
def read_root():
    return {"message": "Welcome to the Deepfake Voice Detector Backend API. Send a POST request with an audio file to /predict."}

@app.post("/predict")
async def analyze_audio(file: UploadFile = File(...)):
    # 1. Validate the file extension
    valid_extensions = ('.flac', '.wav', '.mp3')
    if not file.filename.lower().endswith(valid_extensions):
        raise HTTPException(status_code=400, detail=f"Invalid file type. Please upload one of: {valid_extensions}")

    # 2. Save the uploaded file temporarily so librosa can read it
    temp_file_path = f"temp_{file.filename}"
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. Preprocess the audio
        processed_audio = load_and_preprocess(temp_file_path)

        # 4. Predict
        prediction = model.predict(processed_audio, verbose=0)
        
        # 5. Extract probabilities
        spoof_prob = float(prediction[0][0] * 100)
        real_prob = float(prediction[0][1] * 100)
        
        is_deepfake = spoof_prob > real_prob

        # 6. Return standard JSON response to the frontend
        return {
            "filename": file.filename,
            "human_probability_percent": round(real_prob, 2),
            "fake_probability_percent": round(spoof_prob, 2),
            "verdict": "DEEPFAKE" if is_deepfake else "REAL"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # 7. Always delete the temporary file after processing to save disk space
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# ==========================================
# --- 5. START SERVER ---
# ==========================================
# To run this script locally: `uvicorn server:app --reload`
# Or simply run: `python server.py`
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
