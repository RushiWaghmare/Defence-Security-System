import joblib
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.preprocessing import LabelEncoder

# Load pre-trained VGG16 model without top (fully connected) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Define function to extract features from an image
def extract_features(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Use pre-trained model to extract features
    features = base_model.predict(img_array)

    return features.flatten()  # Flatten the feature tensor to a 1D array

# Load the trained classifier from disk
classifier = joblib.load('svm_model.joblib')

# Directory containing your test images
test_dir = r'C:/Desktop/IOT_Pro/test'

# List all image files in the test directory
test_image_files = [os.path.join(test_dir, file) for file in os.listdir(test_dir) if file.endswith('.jpg') or file.endswith('.png')]

# Extract features for each test image
test_features = [extract_features(image_file) for image_file in test_image_files]

# Convert test features to numpy array
test_features = np.array(test_features)

# Predict on the test set
y_pred_test = classifier.predict(test_features)

# Define class labels for decoding
class_labels = {0: 'aeroplane', 1: 'birds', 2: 'drone', 3: 'helicaptor', 4: 'uav'}

# Decode the predicted labels
decoded_labels = [class_labels[label] for label in y_pred_test]

print("Predicted labels for test images:")
for i, image_file in enumerate(test_image_files):
    print(f"{image_file}: {decoded_labels[i]}")
