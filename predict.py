import os
import sys
import numpy as np
import librosa
import tensorflow as tf

# ==========================================
# --- 1. SETTINGS AND PATHS ---
# ==========================================
# Suppress low-level TensorFlow system logs to keep the terminal clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Path to the trained model we created during training
MODEL_PATH = r"e:\DeepFake_Voice\asv5_detector.h5"

# Audio parameters (MUST be identical to the ones used during training)
SR = 16000          # 16kHz Sample Rate
DURATION = 4        # 4 Seconds
N_MELS = 128        # 128 frequency bands
MAX_TIME_STEPS = 128 # 128 time units

# ==========================================
# --- 2. PREPROCESSING FUNCTION ---
# ==========================================
# This function converts a live audio file into the exact format the AI expects.
def load_and_preprocess(file_path):
    print(f"🔄 Processing file: '{file_path}'...")
    try:
        # Load audio using librosa
        audio, _ = librosa.load(file_path, sr=SR, duration=DURATION)
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        sys.exit(1)

    # STEP A: Pad short files with silence to reach exactly 4 seconds
    if len(audio) < SR * DURATION:
        padding = SR * DURATION - len(audio)
        audio = np.pad(audio, (0, padding), 'constant')

    # STEP B: Convert sound waves to "Mel Spectrogram" (Voice image)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS)
    # Convert power to decibels (log scale) to match training data
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # STEP C: Resize image to exactly 128x128
    if mel_spectrogram.shape[1] < MAX_TIME_STEPS:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, MAX_TIME_STEPS - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :MAX_TIME_STEPS]

    # STEP D: Reshape for the CNN (1, 128, 128, 1) -> (Batch, Height, Width, Channels)
    # The AI expects a "batch" of images, even if we only give it one.
    return np.expand_dims(np.expand_dims(mel_spectrogram, axis=-1), axis=0)

# ==========================================
# --- 3. PREDICTION ENGINE ---
# ==========================================
def predict_audio(file_path):
    # Check if the model exists first
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model file not found at {MODEL_PATH}.")
        print("Please train your model first using 'train_asv5.py'.")
        sys.exit(1)
        
    print("🧠 Loading AI brain (Trained Model)...")
    # Load the H5 model file
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Preprocess the user's specific audio file
    processed_audio = load_and_preprocess(file_path)
    
    print("🔍 Analyzing voice artifacts...")
    # Get the raw prediction numbers (probabilities)
    prediction = model.predict(processed_audio, verbose=0)
    
    # Mapping our training labels:
    # Index 0 = 'spoof' (AI Generated)
    # Index 1 = 'bonafide' (Human Voice)
    spoof_prob = prediction[0][0] * 100
    real_prob = prediction[0][1] * 100
    
    # Print the visual report
    print("\n" + "═"*40)
    print("       📊 DETECTION REPORT       ")
    print("═"*40)
    print(f"  HUMAN VOICE (Prob):  {real_prob:.2f}%")
    print(f"  AI DEEPFAKE (Prob):  {spoof_prob:.2f}%")
    print("═"*40)
    
    # Give the final verdict based on which probability is higher
    if real_prob > spoof_prob:
        print("\n  ✅ VERDICT: This audio is likely REAL.")
    else:
        print("\n  🚨 VERDICT: WARNING! This audio is a DEEPFAKE.")
    print("═"*40 + "\n")

# ==========================================
# --- 4. SCRIPT ENTRY POINT ---
# ==========================================
if __name__ == "__main__":
    # Ensure a file path was provided in the command line
    if len(sys.argv) < 2:
        print("💡 Usage: python predict.py <path_to_audio_file>")
    else:
        audio_file = sys.argv[1]
        if not os.path.exists(audio_file):
            print(f"❌ File not found: {audio_file}")
        else:
            predict_audio(audio_file)
