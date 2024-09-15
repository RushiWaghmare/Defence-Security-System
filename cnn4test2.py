import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import os

# Load pre-trained VGG16 model without top (fully connected) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Convert the VGG16 model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('vgg16_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Define function to extract features from an image
def extract_features(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='vgg16_model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Directory containing your test images
test_dir = r'C:/Desktop/IOT_Pro/test'

# List all image files in the test directory
test_image_files = [os.path.join(test_dir, file) for file in os.listdir(test_dir) if file.endswith('.jpg') or file.endswith('.png')]

# Perform inference on the test set
print("Predicted labels for test images:")
for image_file in test_image_files:
    input_data = extract_features(image_file)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)
    print(f"{image_file}: {predicted_class}")
