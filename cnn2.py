import os
import numpy as np
from PIL import Image
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Define constants
img_width, img_height = 100, 100
input_shape = (img_width, img_height, 3)  # 3 for RGB images
num_classes = 5
epochs = 20
batch_size = 32

# Define paths
train_data_dir = 'C:/Desktop/IOT_Pro/training_set'
model_file = 'image_classifier_model.h5'

# Check if the pre-trained model exists
if os.path.exists(model_file):
    # Load the pre-trained model
    model = load_model(model_file)
    print("Pre-trained model loaded successfully!")
else:
    # Preprocess and augment data
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

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

    # Save the trained model
    model.save(model_file)
    print("Model trained and saved successfully!")

# Class labels
class_labels = ['Aeroplane', 'Drone', 'Helicopter', 'Malicious UAV', 'Other']

# Test the model on a sample image
test_img_path = 'C:/Desktop/IOT_Pro/test/84.png'
test_img = Image.open(test_img_path)
test_img = test_img.resize((img_width, img_height))
test_img_array = img_to_array(test_img)
test_img_array = np.expand_dims(test_img_array, axis=0)  # Add batch dimension
test_img_array /= 255.  # Rescale to [0, 1]

# Predict class probabilities
predictions = model.predict(test_img_array)
predicted_class_index = np.argmax(predictions)
predicted_class = class_labels[predicted_class_index]
print("Predicted Class:", predicted_class)

if predicted_class_index in [0, 1, 2, 3]:
    print("Danger...")
