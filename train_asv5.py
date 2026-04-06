import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
import logging

# ==========================================
# --- 1. CONFIGURATION AND HYPERPARAMETERS ---
# ==========================================
DATA_PATH = r"e:\DeepFake_Voice\data\flac_T"                # Path to flac audio files
METADATA_PATH = r"e:\DeepFake_Voice\data\protocols\ASVspoof5.train.tsv" # Path to labels
MODEL_SAVE_PATH = "asv5_detector.h5"                        # Where to save the trained "brain"

# Audio Processing Settings
SR = 16000          # Standard Sample Rate for voice (16kHz)
DURATION = 4        # We analyze 4-second snippets of audio
N_MELS = 128        # Number of frequency bands (rows in our "image")
MAX_TIME_STEPS = 128 # Fixed width for our "image" (time steps)

# Training Settings
BATCH_SIZE = 16     # Number of files processed at once (Keeps RAM usage low)
EPOCHS = 20         # Max training cycles

logging.basicConfig(level=logging.INFO)

# ==========================================
# --- 2. THE DATA LOADER (MEMORY OPTIMIZER) ---
# ==========================================
# This class acts like a "conveyor belt", loading only 16 files at a time 
# instead of thousands, which prevents your laptop from crashing.
class ASV5DataGenerator(Sequence):
    def __init__(self, file_paths, labels, batch_size=32, dim=(128, 128, 1), n_classes=2, shuffle=True):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Calculate how many batches are in the dataset
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        # Grab the file paths for the current batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_paths = [self.file_paths[k] for k in indexes]
        batch_y = [self.labels[k] for k in indexes]

        # Process the audio files into "images" (spectrograms)
        X, y = self.__data_generation(batch_paths, batch_y)
        return X, y

    def on_epoch_end(self):
        # Shuffle the data after every epoch to help the model learn better
        self.indexes = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_paths, batch_y):
        X = np.empty((self.batch_size, *self.dim)) # Container for audio images
        y = np.empty((self.batch_size), dtype=int) # Container for labels

        for i, (path, label) in enumerate(zip(batch_paths, batch_y)):
            try:
                # STEP A: Load the raw .flac audio
                audio, _ = librosa.load(path, sr=SR, duration=DURATION)
                
                # STEP B: Pad if audio is too short (shorter than 4 seconds)
                if len(audio) < SR * DURATION:
                    padding = SR * DURATION - len(audio)
                    audio = np.pad(audio, (0, padding), 'constant')
                
                # STEP C: Convert sound to "Mel Spectrogram" (Voice Fingerprint image)
                mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS)
                mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
                
                # STEP D: Ensure fixed size (128x128)
                if mel_spectrogram.shape[1] < MAX_TIME_STEPS:
                    mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, MAX_TIME_STEPS - mel_spectrogram.shape[1])), mode='constant')
                else:
                    mel_spectrogram = mel_spectrogram[:, :MAX_TIME_STEPS]

                X[i,] = np.expand_dims(mel_spectrogram, axis=-1)
                # Assign Label: index 1 is Real (Bonafide), index 0 is Fake (Spoof)
                y[i] = 1 if label == 'bonafide' else 0
            except Exception as e:
                logging.warning(f"Error loading {path}: {e}")
                X[i,] = np.zeros(self.dim)
                y[i] = 0

        # One-hot encode the labels (0 -> [1, 0], 1 -> [0, 1])
        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

# ==========================================
# --- 3. METADATA PREPARATION ---
# ==========================================
def prepare_data():
    logging.info("Reading metadata...")
    # ASVspoof 5 format is space-separated without a header
    # index 1: Filename, index 8: spoof/bonafide label
    df = pd.read_csv(METADATA_PATH, sep='\s+', header=None)
    
    # Map filenames to their full system path
    df['full_path'] = df[1].apply(lambda x: os.path.join(DATA_PATH, f"{x}.flac"))
    
    # Only keep entries where the audio file actually exists on your computer
    existing_files = df[df['full_path'].apply(os.path.exists)].copy()
    logging.info(f"Metadata has {len(df)} entries. Found {len(existing_files)} files in your directory.")

    if len(existing_files) == 0:
        raise FileNotFoundError(f"No files found in {DATA_PATH}. Check your dataset path!")

    paths = existing_files['full_path'].tolist()
    labels = existing_files[8].tolist()
    
    return paths, labels

# ==========================================
# --- 4. THE CNN MODEL ARCHITECTURE ---
# ==========================================
def build_model():
    model = models.Sequential([
        # Input Layer: Takes our 128x128 voice fingerprint
        layers.Input(shape=(N_MELS, MAX_TIME_STEPS, 1)),
        
        # Convolutional Block 1: Detects edges and frequency spikes
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2), # Dropout prevents the model from "memorizing" instead of "learning"
        
        # Convolutional Block 2: Detects complex shapes and AI glitch patterns
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Transition from "Picture" to "Numbers"
        layers.Flatten(),
        
        # Dense Layer: The final logic check
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        
        # Output Layer: Probabilities for [FAKE, REAL]
        layers.Dense(2, activation='softmax')
    ])
    
    # Compile with Adam optimizer and CrossEntropy loss (Standard for classification)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# --- 5. MAIN EXECUTION (START TRAINING) ---
# ==========================================
if __name__ == "__main__":
    # Get file paths and their labels
    paths, labels = prepare_data()
    
    # Split: 90% for training, 10% for testing (Validation)
    X_train, X_val, y_train, y_val = train_test_split(paths, labels, test_size=0.1, random_state=42)

    # Initialize the generators
    train_gen = ASV5DataGenerator(X_train, y_train, batch_size=BATCH_SIZE)
    val_gen = ASV5DataGenerator(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

    # Create the model
    model = build_model()
    model.summary() # Print the architecture to the terminal

    logging.info("Starting training process...")
    
    # Start Training!
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[
            # STOP early if accuracy doesn't improve for 5 epochs
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            # SAVE the best version automatically
            tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
        ]
    )
    logging.info(f"Training complete. Best model saved to: {MODEL_SAVE_PATH}")
