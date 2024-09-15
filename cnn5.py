import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load pre-trained VGG16 model without top (fully connected) layers
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Define function to extract features from an image
def extract_features(image_path):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

    # Use pre-trained model to extract features
    features = base_model.predict(img_array)

    return features.flatten()  # Flatten the feature tensor to a 1D array

# Directory containing your images
main_dir = 'C:/Desktop/IOT_Pro/training_set'
subfolders = os.listdir(main_dir)

# List all image files in the directory
image_files = []
all_labels = []
for subfolder in subfolders:
    subfolder_path = os.path.join(main_dir, subfolder)
    if os.path.isdir(subfolder_path):
        label = subfolder
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                image_files.append(os.path.join(subfolder_path, file_name))
                all_labels.append(label)

# Extract features for each image
all_features = [extract_features(image_file) for image_file in image_files]

# Convert features and labels to numpy arrays
all_features = np.array(all_features)
all_labels = np.array(all_labels)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(all_labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_features, encoded_labels, test_size=0.2, random_state=42)

# Define and train a classifier (e.g., Support Vector Machine)
classifier = SVC(kernel='linear', C=1.0, random_state=42)
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Convert the SVM classifier to TensorFlow model
model = Sequential([
    Dense(len(set(encoded_labels)), activation='softmax', input_shape=(X_train.shape[1],))  # Adjust output units based on the number of classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy after conversion to TensorFlow model:", accuracy)

# Convert the TensorFlow model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('svm_model.tflite', 'wb') as f:
    f.write(tflite_model)
