# DeepFake Voice Detection (ASVspoof 5)

This project implements a Convolutional Neural Network (CNN) to detect AI-generated voice deepfakes (spoofs) versus genuine human voices (bonafide). It is built using TensorFlow and Keras, and trained on the modern **ASVspoof 5** challenge dataset.

## 🌟 Key Features
- **Memory-Efficient Training:** Uses a custom `Sequence` DataGenerator to load audio files in small batches (16 at a time), allowing the model to be trained on standard consumer laptops without crashing due to RAM limitations.
- **Audio Fingerprinting:** Converts raw `.flac` audio waves into **Mel Spectrograms** (128x128 images) using `librosa`, allowing the CNN to visually scan for unnatural frequency anomalies typical of AI generation.
- **Early Stopping & Checkpointing:** Automatically monitors validation accuracy during training. It saves the best-performing model to disk (`asv5_detector.h5`) and stops early if the model stops improving, saving time and compute resources.
- **Ready-to-use Inference Script:** Includes a `predict.py` script to easily test any new audio file against the trained model and get a clear "REAL vs FAKE" percentage breakdown.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Deep Learning:** TensorFlow 2.x, Keras
- **Audio Processing:** Librosa, SoundFile
- **Data Manipulation:** NumPy, Pandas, Scikit-learn

## 📁 Repository Structure
```text
DeepFake_Voice/
├── data/                       # (Ignored in Git due to size)
│   ├── protocols/              # Contains ASVspoof5.train.tsv (Labels)
│   └── flac_T/                 # Contains raw training audio (.flac files)
├── train_asv5.py               # Main training script (Memory Optimized)
├── predict.py                  # Inference script to test single audio files
├── requirements.txt            # Project dependencies
├── .gitignore                  # Prevents pushing massive datasets and local envs 
└── asv5_detector.h5            # The trained AI model (Generated after training)
```

## 🧠 Model Architecture
The internal "brain" is a 2D Convolutional Neural Network (CNN) specifically tuned for image-based audio classification:
1.  **Input:** $128 \times 128 \times 1$ Mel Spectrogram image (representing 4 seconds of audio).
2.  **Conv Block 1:** `Conv2D` (32 filters, 3x3) $\rightarrow$ `MaxPooling2D` $\rightarrow$ `Dropout` (0.2)
3.  **Conv Block 2:** `Conv2D` (64 filters, 3x3) $\rightarrow$ `MaxPooling2D` $\rightarrow$ `Dropout` (0.2)
4.  **Classification Head:** `Flatten` $\rightarrow$ `Dense` (128 units, ReLU) $\rightarrow$ `Dropout` (0.5)
5.  **Output Layer:** `Dense` (2 units, Softmax) predicting probabilities for `[FAKE, REAL]`.

## 🚀 Getting Started

### 1. Setup Environment
It is highly recommended to use a virtual environment.
```powershell
# Create and activate your virtual environment (Windows)
python -m venv venv
.\venv\Scripts\activate

# Install all necessary libraries
pip install -r requirements.txt
```

### 2. Data Preparation
This project is configured for the **ASVspoof 5** dataset. 
- Place your unzipped training audio files inside `data/flac_T/`.
- Place your label metadata file (`ASVspoof5.train.tsv`) inside `data/protocols/`.

### 3. Training the Model
Run the training script. The console will display a progress bar and automatically save the best outcome as `asv5_detector.h5`.
```powershell
python train_asv5.py
```
*(You can manually interrupt training `Ctrl+C` early if the validation accuracy reaches a satisfactory level (e.g., > 90%); the prior best epoch will have already been saved.)*

### 4. Testing / Predicting
Once you have trained the model (or if you already have the `asv5_detector.h5` file generated), you can scan any audio file to see if it is a deepfake:
```powershell
python predict.py path\to\your\audiofile.flac
```
**Example Output:**
```text
🔄 Processing file: 'data\flac_T\T_0000000000.flac'...
🧠 Loading AI brain (Trained Model)...
🔍 Analyzing voice artifacts...

════════════════════════════════════════
       📊 DETECTION REPORT       
════════════════════════════════════════
  HUMAN VOICE (Prob):  5.12%
  AI DEEPFAKE (Prob):  94.88%
════════════════════════════════════════

  🚨 VERDICT: WARNING! This audio is a DEEPFAKE.
════════════════════════════════════════
```

## 📊 Dataset Reference
- **Dataset:** ASVspoof 5 (2024 Track)
- **Format:** `.flac` audio files (16kHz)
- **Source:** [University of Edinburgh Datashare - ASVspoof 5](https://datashare.ed.ac.uk/handle/10283/3336)
