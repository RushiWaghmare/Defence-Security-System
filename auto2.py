import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Function to load and preprocess the dataset for image enhancement
def load_and_preprocess_dataset(directory, image_size):
    X = []
    for filename in os.listdir(directory):
        img = load_img(os.path.join(directory, filename), target_size=image_size)
        img_array = img_to_array(img)
        X.append(img_array)
    X = np.array(X) / 255.0  # Normalize pixel values to range [0, 1]
    return X

# Define the directory containing the dataset for image enhancement
directory = 'C:/Desktop/IOT_Pro/gandataset/data'

# Define the image size for training
image_size = (128, 128)  # Adjust image size as needed

# Load and preprocess the dataset for image enhancement
X = load_and_preprocess_dataset(directory, image_size)

# Define the autoencoder architecture for image enhancement
input_img = Input(shape=image_size + (3,))

# Define encoder layers
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Define decoder layers
x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# Create the autoencoder model
autoencoder = Model(input_img, decoded)

# Compile the autoencoder
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

# Train-test split
X_train, X_val = train_test_split(X, test_size=0.2)

# Train the autoencoder for image enhancement
history = autoencoder.fit(X_train, X_train,
                           epochs=20,
                           batch_size=32,
                           shuffle=True,
                           validation_data=(X_val, X_val))

# Function to enhance an input image using the trained autoencoder
def enhance_image(autoencoder, input_image):
    enhanced_image = autoencoder.predict(np.expand_dims(input_image, axis=0))
    return enhanced_image[0]

# Example usage: Enhance an input image
input_image_path = 'C:/Desktop/IOT_Pro/gandataset/test.jpg'
input_image = load_img(input_image_path, target_size=image_size)
input_image_array = img_to_array(input_image) / 255.0  # Normalize pixel values
enhanced_image_array = enhance_image(autoencoder, input_image_array)

# Display the original and enhanced images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(input_image)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('Enhanced Image')
plt.imshow(enhanced_image_array)
plt.axis('off')
plt.show()

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
