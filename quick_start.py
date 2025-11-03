#!/usr/bin/env python3
"""
Quick Start Script for CNN Emotion Recognition
This script helps you get started quickly with the project
"""

import os
import sys

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def check_files():
    """Check if necessary files exist"""
    files = {
        'cnn_model.py': 'CNN model architecture',
        'train_model.py': 'Training script',
        'emotion_cnn.py': 'Real-time detection script',
        'requirements.txt': 'Python dependencies'
    }
    
    print("Checking project files...")
    all_exist = True
    for file, desc in files.items():
        exists = os.path.exists(file)
        status = "✓" if exists else "✗"
        print(f"  {status} {file:<20} - {desc}")
        if not exists:
            all_exist = False
    
    return all_exist

def check_dataset():
    """Check if dataset exists"""
    print("\nChecking for dataset...")
    
    if os.path.exists('fer2013.csv'):
        print("  ✓ fer2013.csv found")
        return True
    elif os.path.exists('fer2013') and os.path.isdir('fer2013'):
        print("  ✓ fer2013 directory found")
        return True
    else:
        print("  ✗ Dataset not found")
        print("\n  Please download FER2013 dataset:")
        print("  1. Go to: https://www.kaggle.com/datasets/msambare/fer2013")
        print("  2. Download fer2013.csv")
        print("  3. Place it in this directory")
        return False

def check_model():
    """Check if trained model exists"""
    print("\nChecking for trained model...")
    
    if os.path.exists('best_emotion_model.h5'):
        print("  ✓ best_emotion_model.h5 found")
        return True
    elif os.path.exists('emotion_cnn_model.h5'):
        print("  ✓ emotion_cnn_model.h5 found")
        return True
    else:
        print("  ✗ No trained model found")
        print("  You need to train a model first")
        return False

def main():
    print_header("CNN Emotion Recognition - Quick Start")
    
    # Check files
    if not check_files():
        print("\n❌ Some project files are missing!")
        print("Please ensure all files are in the directory.")
        sys.exit(1)
    
    # Check dataset
    has_dataset = check_dataset()
    
    # Check model
    has_model = check_model()
    
    # Provide next steps
    print_header("Next Steps")
    
    if not has_dataset and not has_model:
        print("📥 STEP 1: Download Dataset")
        print("   Visit: https://www.kaggle.com/datasets/msambare/fer2013")
        print("   Download fer2013.csv and place it here\n")
        print("🔧 STEP 2: Install Dependencies")
        print("   Run: pip install -r requirements.txt\n")
        print("🎓 STEP 3: Train the Model")
        print("   Run: python train_model.py\n")
        print("🎥 STEP 4: Run Real-time Detection")
        print("   Run: python emotion_cnn.py\n")
        
    elif has_dataset and not has_model:
        print("✓ Dataset found!\n")
        print("🔧 STEP 1: Install Dependencies (if not done)")
        print("   Run: pip install -r requirements.txt\n")
        print("🎓 STEP 2: Train the Model")
        print("   Run: python train_model.py")
        print("   (This will take 30-60 minutes)\n")
        print("🎥 STEP 3: Run Real-time Detection")
        print("   Run: python emotion_cnn.py\n")
        
    elif has_model:
        print("✓ Trained model found!\n")
        print("🎥 You're ready to run real-time detection!")
        print("   Run: python emotion_cnn.py\n")
        print("Optional:")
        print("  - View model architecture: python cnn_model.py")
        print("  - Retrain model: python train_model.py\n")
    
    # Additional help
    print_header("Additional Information")
    print("📖 Full documentation: See README_CNN.md")
    print("🎨 Model architecture: Run python cnn_model.py")
    print("⚙️  Training options: Edit train_model.py")
    print("\n🔗 Useful Links:")
    print("   Dataset: https://www.kaggle.com/datasets/msambare/fer2013")
    print("   TensorFlow: https://www.tensorflow.org/")
    print("   Keras: https://keras.io/\n")
    
    print_header("Project Structure")
    print("""
    cnn_model.py       → Defines CNN architecture
    train_model.py     → Trains the model on FER2013
    emotion_cnn.py     → Real-time emotion detection
    emotion.py         → Original (DeepFace version)
    """)

if __name__ == "__main__":
    main()
