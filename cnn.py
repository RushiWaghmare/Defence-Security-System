import joblib
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input  # type: ignore

BASE_DIR = "C:/Desktop/IOT_Pro/api"
CLASS_LABELS = {0: "aeroplane", 1: "birds", 2: "drone", 3: "helicaptor", 4: "uav"}
# Load pre-trained VGG16 model without top (fully connected) layers
BASE_MODEL = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))


# Define function to extract features from an image
def extract_features(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Use pre-trained model to extract features
    features = BASE_MODEL.predict(img_array)

    return features.flatten()  # Flatten the feature tensor to a 1D array


# Load the trained classifier from disk
classifier = joblib.load("svm_model.joblib")


def run_cnn(image_path: str):
    image_path = os.path.join(BASE_DIR, image_path)
    image_features = np.array([extract_features(image_path)])
    predictions = classifier.predict(image_features)
    decoded_labels = [CLASS_LABELS[label] for label in predictions]
    print(decoded_labels)
    return decoded_labels


if __name__ == "__main__":
    # Directory containing your test images
    test_dir = r"C:/Desktop/IOT_Pro/test"

    # List all image files in the test directory
    test_image_files = [
        os.path.join(test_dir, file)
        for file in os.listdir(test_dir)
        if file.endswith(".jpg") or file.endswith(".png")
    ]

    # Extract features for each test image
    run_cnn('C:/Desktop/IOT_Pro/test/image2_Low_quality.jpg')
    # test_features = [extract_features(image_file) for image_file in test_image_files]
    # test_features = [run_cnn(image_file) for image_file in test_image_files]
    exit(0)

    # Convert test features to numpy array
    test_features = np.array(test_features)

    # # Predict on the test set
    y_pred_test = classifier.predict(test_features)

    # Decode the predicted labels
    decoded_labels = [CLASS_LABELS[label] for label in y_pred_test]

    print("Predicted labels for test images:")
    for i, image_file in enumerate(test_image_files):
        print(f"{image_file}: {decoded_labels[i]}")
