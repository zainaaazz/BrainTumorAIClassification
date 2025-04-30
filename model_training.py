#importing all the required libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

#Step 1: definnig my paths to the data
train_path = "data/Training"
test_path = "data/Testing"

# ==== Class Distribution Count and Visualization ====
# Function to Count Images
def get_class_distribution(path):
    classes = os.listdir(path)
    data = []
    for class_name in classes:
        class_path = os.path.join(path, class_name)
        count = len(os.listdir(class_path))
        data.append({'Tumor_Type': class_name, 'Image_Count': count})
    return pd.DataFrame(data)

# Training Data Class Count
train_df = get_class_distribution(train_path)
print("Training Data Class Distribution:")
print(train_df)

# Testing Data Class Count
test_df = get_class_distribution(test_path)
print("\nTesting Data Class Distribution:")
print(test_df)

# Plotting the class distribution
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.barplot(x='Tumor_Type', y='Image_Count', data=train_df)
plt.title('Training Data Distribution')

plt.subplot(1,2,2)
sns.barplot(x='Tumor_Type', y='Image_Count', data=test_df)
plt.title('Testing Data Distribution')

plt.tight_layout()
plt.show()

# sTEP 2 : data augmentation & Preprocessingg -creates "artificial" variations of existing images to help the model generalise better
#flipping, zooming or slightly rotating images to simulate different scenarios
train_datagen = ImageDataGenerator(
    rescale=1./255, #normalizsing pixel values from 0–255 to 0–1
    shear_range=0.2, #Applying shearing (slanting) transformation
    zoom_range=0.2, #random zooming inside the image
    horizontal_flip=True  #randomly flipping some images left-right
)
test_datagen = ImageDataGenerator(rescale=1./255)   # Only normalise test data (no augmentation)

train = train_datagen.flow_from_directory(
    train_path,
    target_size=(128, 128), #resizing all images to 128x128 pixels
    batch_size=32, #process 32 images at a time
    class_mode='categorical'   # coz this is a multi-class classification  w/ 4 tumor types
)
test = test_datagen.flow_from_directory(
    test_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

#Step 3 - MODEL ARCHITECTURE - Building the Convolutional Neural Network (CNN)
# CNNs are designed to work well with image data
#They automatically learn to detect patterns like edges, shapes, textures, and then more complex features
model = Sequential([

    #1st convolutional layer: applies 32 filters (3x3) and ReLU activation
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2), # Downsample the image to reduce complexity

    # 2nd conv + pooling
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    #3rd Conv + pooling
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    #flatten the 3D output to 1D for the fully connected layers
    Flatten(),

    # Fully connected layer with 128 neurons
    Dense(128, activation='relu'),

     #dropout layer to  randomly "turns off" some neurons to prevent overfitting
    Dropout(0.5),

     #output layer: 4 neurons for the 4 tumor types, softmax gives probabilities
    Dense(4, activation='softmax')  # 4 output classes
])

# STEP 4: Compiling the the Model 
# Before training, we define:
# - optimizer: how the model improves itself (Adam is a popular choice)
# - loss function: how it measures "error"
# - metrics: wE want to track accuracy in this case

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# STERP 5 Training the Model 
#fits (trains) the model using your data
# can adjust the number of `epochs` (how many times the model sees the entire dataset)

#Step 6: Add Early Stopping 
early_stop = EarlyStopping(
    monitor='val_loss',       # Watch validation loss
    patience=3,               # Stop if no improvement after 3 epochs
    restore_best_weights=True,  # Roll back to best-performing model
    verbose=1  # Prints when it stops early
)

# Step 7 train
history = model.fit(
    train,
    validation_data=test,
    epochs=20,                # Go up to 20, but early stopping will kick in
    callbacks=[early_stop]
)

#step 8  Save 
#allow to reuse the trained model in Streamlit/other applications
#saves the model architecture + learned weights in a file

if not os.path.exists("model"):
    os.makedirs("model")
model.save("model/brain_tumor_model.h5")
print("Model saved to model/brain_tumor_model.h5")

# ==== Plot Accuracy ====
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='x')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# ==== Plot Loss ====
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='x')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Show both plots
plt.tight_layout()
plt.show()

# ==== Evaluate Final Model Performance ====
score = model.evaluate(test)
print('\nFinal Evaluation on Test Data:')
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])
