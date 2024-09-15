import os
import numpy as np
from PIL import Image
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
img_width, img_height = 100, 100
input_shape = (img_width, img_height, 3)  # 3 for RGB images
num_classes = 5
epochs = 20
batch_size = 32

# Define paths
train_data_dir = 'C:/Desktop/IOT_Pro/training_set'
cnn = 'image_classifier_model.h5'

# Check if model file exists
if os.path.exists(cnn):
    # Load the saved model
    model = load_model(cnn)
    print("Model loaded successfully.")
else:
    print("Model file not found. Please train the model first.")

# Test the model on a sample image
test_img_path = 'C:/Desktop/IOT_Pro/test/31.png'
test_img = Image.open(test_img_path)
test_img = test_img.resize((img_width, img_height))
test_img = np.expand_dims(test_img, axis=0)  # Add batch dimension

# Predict class probabilities
predictions = model.predict(test_img)
class_labels = ['Aeroplane', 'Drone', 'Helicopter', 'Malicious UAV',]
predicted_class = class_labels[np.argmax(predictions)]
print("Predicted Class:", predicted_class)

if predicted_class in class_labels:
    print("Danger...")
