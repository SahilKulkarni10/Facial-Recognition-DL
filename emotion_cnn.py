"""
Real-time Emotion Detection using Custom CNN Model
This uses a trained CNN model instead of DeepFace
"""

import cv2
import numpy as np
from keras.models import load_model
import sys
import time
import os

# Emotion labels (same order as training)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define colors for different emotions (BGR format)
EMOTION_COLORS = {
    'Angry': (0, 0, 255),       # Red
    'Disgust': (0, 128, 128),   # Dark Yellow
    'Fear': (255, 0, 255),      # Magenta
    'Happy': (0, 255, 0),       # Green
    'Sad': (255, 0, 0),         # Blue
    'Surprise': (0, 255, 255),  # Yellow
    'Neutral': (255, 255, 255)  # White
}

class EmotionDetector:
    def __init__(self, model_path='emotion_cnn_model.h5'):
        """Initialize the emotion detector with trained CNN model"""
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found!")
            print("\nPlease train the model first:")
            print("1. Download FER2013 dataset")
            print("2. Run: python train_model.py")
            print("\nAlternatively, you can download a pre-trained model.")
            sys.exit(1)
        
        # Load the trained CNN model
        print(f"Loading CNN model from {model_path}...")
        self.model = load_model(model_path)
        print("Model loaded successfully!")
        
        # Load face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def preprocess_face(self, face_img):
        """
        Preprocess face image for CNN model
        
        Args:
            face_img: Face ROI from video frame
            
        Returns:
            Preprocessed image ready for model prediction
        """
        # Resize to 48x48 (model input size)
        face_img = cv2.resize(face_img, (48, 48))
        
        # Convert to grayscale if not already
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Normalize pixel values
        face_img = face_img / 255.0
        
        # Reshape for model input (batch_size, height, width, channels)
        face_img = face_img.reshape(1, 48, 48, 1)
        
        return face_img
    
    def predict_emotion(self, face_roi):
        """
        Predict emotion from face ROI using CNN model
        
        Args:
            face_roi: Face region of interest
            
        Returns:
            Tuple of (emotion_label, confidence, all_probabilities)
        """
        # Preprocess the face
        processed_face = self.preprocess_face(face_roi)
        
        # Predict using CNN model
        predictions = self.model.predict(processed_face, verbose=0)[0]
        
        # Get the emotion with highest probability
        emotion_idx = np.argmax(predictions)
        emotion = EMOTIONS[emotion_idx]
        confidence = predictions[emotion_idx]
        
        return emotion, confidence, predictions
    
    def detect_and_display(self):
        """Main loop for real-time emotion detection"""
        
        # Start capturing video
        cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not access the camera.")
            print("\nPlease grant camera permissions:")
            print("1. Open System Settings > Privacy & Security > Camera")
            print("2. Enable camera access for Terminal or your Python application")
            print("3. Run the script again")
            sys.exit(1)
        
        # Wait a moment for camera to initialize
        time.sleep(1)
        
        print("Camera initialized successfully!")
        print("Press 'q' to quit the application")
        print("Press 's' to show emotion probabilities")
        
        show_probabilities = False
        
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            # Check if frame was captured successfully
            if not ret or frame is None:
                print("Error: Failed to capture frame from camera")
                break
            
            # Convert frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the frame
            faces = self.face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces:
                # Extract the face ROI (Region of Interest)
                face_roi = gray_frame[y:y + h, x:x + w]
                
                # Predict emotion using CNN model
                emotion, confidence, all_probs = self.predict_emotion(face_roi)
                
                # Get color for the emotion
                color = EMOTION_COLORS.get(emotion, (255, 255, 255))
                
                # Draw rectangle around face with emotion color
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                
                # Prepare emotion text
                emotion_text = f"{emotion} ({confidence*100:.1f}%)"
                font_scale = 1.0
                thickness = 3
                
                # Add black background behind text for better visibility
                (text_width, text_height), baseline = cv2.getTextSize(
                    emotion_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    thickness
                )
                cv2.rectangle(
                    frame,
                    (x, y - text_height - 20),
                    (x + text_width, y),
                    (0, 0, 0),
                    -1
                )
                
                # Draw the emotion text
                cv2.putText(
                    frame,
                    emotion_text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    thickness
                )
                
                # Show all emotion probabilities if enabled
                if show_probabilities:
                    prob_y = y + h + 30
                    for i, (emo, prob) in enumerate(zip(EMOTIONS, all_probs)):
                        prob_text = f"{emo}: {prob*100:.1f}%"
                        cv2.putText(
                            frame,
                            prob_text,
                            (x, prob_y + i*25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1
                        )
            
            # Add instructions
            cv2.putText(
                frame,
                "Press 'q' to quit | 's' to toggle probabilities",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Display the resulting frame
            cv2.imshow('CNN-Based Emotion Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                show_probabilities = not show_probabilities
        
        # Release the capture and close all windows
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Main function to run the emotion detector"""
    
    print("="*60)
    print("CNN-Based Real-time Emotion Detection")
    print("="*60)
    
    # Check for model files
    if os.path.exists('best_emotion_model.h5'):
        model_path = 'best_emotion_model.h5'
        print("Using best_emotion_model.h5")
    elif os.path.exists('emotion_cnn_model.h5'):
        model_path = 'emotion_cnn_model.h5'
        print("Using emotion_cnn_model.h5")
    else:
        print("\nNo trained model found!")
        print("Please train a model first by running: python train_model.py")
        print("\nOr download a pre-trained model and save it as 'emotion_cnn_model.h5'")
        sys.exit(1)
    
    # Create detector and start detection
    detector = EmotionDetector(model_path)
    detector.detect_and_display()
    
    print("\nEmotion detection stopped.")


if __name__ == "__main__":
    main()
