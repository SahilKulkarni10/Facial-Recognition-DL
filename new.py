import cv2
from deepface import DeepFace
import sys
import time

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame was captured successfully
    if not ret or frame is None:
        print("Error: Failed to capture frame from camera")
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        
        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Determine the dominant emotion
        emotion = result[0]['dominant_emotion']

        # Define colors for different emotions (BGR format)
        emotion_colors = {
            'happy': (0, 255, 0),      # Green
            'sad': (255, 0, 0),         # Blue
            'angry': (0, 0, 255),       # Red
            'surprise': (0, 255, 255),  # Yellow
            'fear': (255, 0, 255),      # Magenta
            'disgust': (0, 128, 128),   # Dark Yellow
            'neutral': (255, 255, 255)  # White
        }
        
        # Get color for the emotion (default to white if not found)
        color = emotion_colors.get(emotion, (255, 255, 255))

        # Draw rectangle around face with emotion color
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        
        # Make emotion text BIGGER and BOLDER
        emotion_text = emotion.upper()  # Make text uppercase
        font_scale = 1.5  # Larger font size
        thickness = 4  # Thicker text
        
        # Add black background behind text for better visibility
        (text_width, text_height), baseline = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(frame, (x, y - text_height - 20), (x + text_width, y), (0, 0, 0), -1)
        
        # Draw the emotion text
        cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

