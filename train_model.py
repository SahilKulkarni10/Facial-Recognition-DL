"""
Training script for CNN-based emotion recognition
Trains on FER2013 dataset or custom dataset
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from cnn_model import create_emotion_cnn, create_simple_emotion_cnn

# Emotion labels
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def load_fer2013_dataset(csv_path='fer2013.csv'):
    """
    Load FER2013 dataset from CSV file
    
    Download from: https://www.kaggle.com/datasets/msambare/fer2013
    Or use: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
    """
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        print("\nPlease download FER2013 dataset:")
        print("1. Go to: https://www.kaggle.com/datasets/msambare/fer2013")
        print("2. Download fer2013.csv")
        print("3. Place it in the project directory")
        return None, None, None, None
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Extract pixels and labels
    pixels = df['pixels'].tolist()
    emotions = df['emotion'].tolist()
    
    # Convert pixels to numpy arrays
    X = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.array(face).reshape(48, 48, 1)
        X.append(face)
    
    X = np.array(X, dtype='float32')
    y = np.array(emotions)
    
    # Normalize pixel values
    X = X / 255.0
    
    # Convert labels to categorical
    y = to_categorical(y, num_classes=7)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Dataset loaded successfully!")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def load_fer2013_from_directory(base_path='fer2013'):
    """
    Load FER2013 dataset from directory structure
    
    Expected structure:
    fer2013/
        train/
            angry/
            disgust/
            ...
        test/
            angry/
            disgust/
            ...
    """
    
    if not os.path.exists(base_path):
        print(f"Error: {base_path} directory not found!")
        return None, None
    
    train_dir = os.path.join(base_path, 'train')
    test_dir = os.path.join(base_path, 'test')
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for test
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=64,
        class_mode='categorical',
        shuffle=True
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=64,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, test_generator


def train_model(model_type='simple', use_directory=False):
    """
    Train the emotion recognition model
    
    Args:
        model_type: 'simple' or 'complex'
        use_directory: True if using directory structure, False for CSV
    """
    
    # Create model
    if model_type == 'simple':
        print("Creating Simple CNN model...")
        model = create_simple_emotion_cnn()
    else:
        print("Creating Complex CNN model...")
        model = create_emotion_cnn()
    
    model.summary()
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        'best_emotion_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # Train model
    if use_directory:
        # Load data from directory
        train_gen, test_gen = load_fer2013_from_directory()
        
        if train_gen is None:
            return None
        
        history = model.fit(
            train_gen,
            validation_data=test_gen,
            epochs=50,
            callbacks=callbacks
        )
    else:
        # Load data from CSV
        X_train, X_test, y_train, y_test = load_fer2013_dataset()
        
        if X_train is None:
            return None
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=64),
            validation_data=(X_test, y_test),
            epochs=50,
            callbacks=callbacks
        )
    
    # Save final model
    model.save('emotion_cnn_model.h5')
    print("\nModel saved as 'emotion_cnn_model.h5'")
    
    # Plot training history
    plot_training_history(history)
    
    return model, history


def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")
    plt.show()


if __name__ == "__main__":
    print("="*60)
    print("CNN Emotion Recognition - Training Script")
    print("="*60)
    
    print("\nOptions:")
    print("1. Train with FER2013 CSV file (fer2013.csv)")
    print("2. Train with FER2013 directory structure (fer2013/train, fer2013/test)")
    print("3. Use simple model (faster, less accurate)")
    print("4. Use complex model (slower, more accurate)")
    
    # For automatic training, use simple model with CSV
    print("\nStarting training with simple model...")
    print("Note: Please download FER2013 dataset first if you haven't already")
    
    model, history = train_model(model_type='simple', use_directory=False)
    
    if model:
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("="*60)
