import os
import numpy as np
import cv2
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam

# Function to load and preprocess images
def load_images(folder_path):
    image_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder_path, filename))
            img = cv2.resize(img, (128, 128))  # Resize images to a consistent size
            img = img.astype('float32') / 255.0  # Normalize pixel values
            image_list.append(img)
    return np.array(image_list)

# Load and preprocess high and low-quality images
high_quality_images = load_images("C:/Desktop/IOT_Pro/gandataset/data2/image1_High_quality")
low_quality_images = load_images("C:/Desktop/IOT_Pro/gandataset/data2/image1_Low_quality")

# Define autoencoder architecture
input_img = Input(shape=(128, 128, 3))

# Encoder
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# Autoencoder model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(low_quality_images, high_quality_images,
                epochs=10,
                batch_size=32,
                shuffle=True,
                validation_split=0.2)
