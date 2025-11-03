# CNN-Based Facial Emotion Recognition

A real-time emotion detection system using a **custom Convolutional Neural Network (CNN)** built with TensorFlow/Keras.

## 🎯 Features

- **Custom CNN Architecture**: Built from scratch with multiple convolutional layers, batch normalization, and dropout
- **Real-time Detection**: Detects emotions from webcam feed in real-time
- **7 Emotion Classes**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Color-coded Visualization**: Different colors for each emotion
- **Confidence Scores**: Shows prediction confidence percentages
- **Probability Display**: Toggle to see all emotion probabilities

## 🏗️ CNN Architecture

### Simple Model (Faster Training)
- 3 Convolutional blocks (32, 64, 128 filters)
- MaxPooling and Dropout layers
- 1 Fully connected layer (256 neurons)
- Output layer with softmax activation

### Complex Model (Higher Accuracy)
- 4 Convolutional blocks (32, 64, 128, 256 filters)
- Batch Normalization after each Conv layer
- MaxPooling and Dropout for regularization
- 2 Fully connected layers (512, 256 neurons)
- Output layer with softmax activation

## 📋 Requirements

```bash
pip install -r requirements.txt
```

## 📦 Dataset

This project uses the **FER2013 dataset** for training.

### Download Options:

1. **Kaggle** (Recommended):
   - Go to: https://www.kaggle.com/datasets/msambare/fer2013
   - Download `fer2013.csv`
   - Place in project directory

2. **Alternative Directory Structure**:
   ```
   fer2013/
       train/
           angry/
           disgust/
           fear/
           happy/
           sad/
           surprise/
           neutral/
       test/
           (same structure)
   ```

## 🚀 Usage

### Step 1: View Model Architecture

```bash
python cnn_model.py
```

This will display the CNN architecture and layer details.

### Step 2: Train the Model

```bash
python train_model.py
```

**Training Parameters:**
- Epochs: 50 (with early stopping)
- Batch size: 64
- Optimizer: Adam (learning rate: 0.0001)
- Data augmentation: Rotation, shifting, flipping
- Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

**Outputs:**
- `best_emotion_model.h5` - Best model during training
- `emotion_cnn_model.h5` - Final trained model
- `training_history.png` - Training/validation accuracy and loss plots

### Step 3: Run Real-time Detection

```bash
python emotion_cnn.py
```

**Controls:**
- Press `q` to quit
- Press `s` to toggle emotion probabilities display

## 📁 Project Structure

```
Facial-Recognition-DL/
│
├── cnn_model.py              # CNN model architecture
├── train_model.py            # Training script
├── emotion_cnn.py            # Real-time detection with CNN
├── emotion.py                # Original DeepFace version (for comparison)
├── requirements.txt          # Python dependencies
├── README_CNN.md            # This file
│
├── fer2013.csv              # Dataset (download separately)
├── emotion_cnn_model.h5     # Trained model (generated after training)
└── training_history.png     # Training plots (generated after training)
```

## 🎨 Emotion Color Coding

| Emotion  | Color   | BGR Value     |
|----------|---------|---------------|
| Happy    | Green   | (0, 255, 0)   |
| Sad      | Blue    | (255, 0, 0)   |
| Angry    | Red     | (0, 0, 255)   |
| Surprise | Yellow  | (0, 255, 255) |
| Fear     | Magenta | (255, 0, 255) |
| Disgust  | Dark Yellow | (0, 128, 128) |
| Neutral  | White   | (255, 255, 255) |

## 🔄 Comparison: CNN vs DeepFace

| Aspect | Custom CNN (emotion_cnn.py) | DeepFace (emotion.py) |
|--------|----------------------------|----------------------|
| **Model** | Your own trained CNN | Pre-trained library model |
| **Training** | Required (train_model.py) | Not needed |
| **Control** | Full control over architecture | Black-box |
| **Customization** | Highly customizable | Limited |
| **Dataset** | FER2013 (or custom) | Pre-trained on various datasets |
| **Speed** | Depends on your model | Generally faster |
| **Learning** | Learn CNN architecture & training | Easy to use |

## 🎓 What Makes This a True CNN Project?

✅ **Custom CNN Architecture**: You define the layers, filters, and connections  
✅ **Training from Scratch**: You train the model on the dataset  
✅ **Hyperparameter Tuning**: You control learning rate, batch size, epochs  
✅ **Data Augmentation**: You implement rotation, shifting, flipping  
✅ **Model Optimization**: Callbacks for best model saving, early stopping  
✅ **Understanding**: You understand every layer and its purpose  

## 📊 Expected Performance

- **Training Accuracy**: ~60-70% (FER2013 is challenging)
- **Validation Accuracy**: ~55-65%
- **Inference Speed**: 20-30 FPS on modern CPU
- **Training Time**: 30-60 minutes (depends on hardware)

## 🔧 Troubleshooting

### Camera Access Issues (macOS)
```
System Settings > Privacy & Security > Camera
Enable for Terminal/Python
```

### Model Not Found
```bash
# Make sure you've trained the model first
python train_model.py
```

### Low Accuracy
- Train for more epochs
- Use the complex model
- Add more data augmentation
- Try different learning rates

## 🚀 Future Improvements

- [ ] Try different CNN architectures (ResNet, VGG, EfficientNet)
- [ ] Use transfer learning with pre-trained models
- [ ] Implement ensemble methods
- [ ] Add attention mechanisms
- [ ] Try other datasets (CK+, AffectNet)
- [ ] Deploy as web application
- [ ] Add age and gender detection

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

Feel free to fork, improve, and submit pull requests!

---

**Created with ❤️ for learning Deep Learning and Computer Vision**
