import os
import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Define constants
img_width, img_height = 100, 100
input_shape = (img_width, img_height, 3)  # 3 for RGB images
num_classes = 5
epochs = 20
batch_size = 32

# Define paths
train_data_dir = r'C:/Desktop/IOT_Pro/training_set'

# Preprocess and augment data
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=epochs)

# Save the model
model.save('image_classifier_model.h5')

# Class labels
class_labels = list(train_generator.class_indices.keys())
print("Class Labels:", class_labels)

# Test the model on a sample image
test_img_path = 'C:/Desktop/IOT_Pro/test/31.png'
test_img = Image.open(test_img_path)
test_img = test_img.resize((img_width, img_height))
test_img = np.expand_dims(test_img, axis=0)  # Add batch dimension

# Predict class probabilities
predictions = model.predict(test_img)
predicted_class = class_labels[np.argmax(predictions)]
print("Predicted Class:", predicted_class)

if predicted_class in ['Aeroplane', 'Drone', 'Helicopter', 'Malicious UAV']:
    print("Danger...")

    
