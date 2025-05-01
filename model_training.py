# === Brain Tumor Classification using Transfer Learning (ResNet50) ===

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# ==== Step 1: Set Paths ====
train_path = "data/Training"
test_path = "data/Testing"

# ==== Step 2: Count Images per Class ====
def get_class_distribution(path):
    classes = os.listdir(path)
    data = []
    for class_name in classes:
        class_path = os.path.join(path, class_name)
        count = len(os.listdir(class_path))
        data.append({'Tumor_Type': class_name, 'Image_Count': count})
    return pd.DataFrame(data)

train_df = get_class_distribution(train_path)
test_df = get_class_distribution(test_path)

print("Training Data Class Distribution:")
print(train_df)
print("\nTesting Data Class Distribution:")
print(test_df)

# ==== Step 3: Plot Class Distribution ====
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.barplot(x='Tumor_Type', y='Image_Count', data=train_df)
plt.title('Training Data Distribution')

plt.subplot(1, 2, 2)
sns.barplot(x='Tumor_Type', y='Image_Count', data=test_df)
plt.title('Testing Data Distribution')
plt.tight_layout()
os.makedirs("model", exist_ok=True)
plt.savefig("model/class_distribution.png")
plt.show()

# ==== Step 4: Data Preprocessing & Augmentation (Improved) ====
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)



train = train_datagen.flow_from_directory(
    train_path, target_size=(224, 224), batch_size=64, class_mode='categorical'
)
test = test_datagen.flow_from_directory(
    test_path, target_size=(224, 224), batch_size=64, class_mode='categorical'
)


# ==== Step 5: Compute Class Weights ====
class_labels = list(train.class_indices.keys())
class_counts = train_df.set_index('Tumor_Type').loc[class_labels]['Image_Count'].to_numpy()
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(len(class_labels)),
    y=np.repeat(np.arange(len(class_labels)), class_counts)
)
class_weights = dict(enumerate(class_weights_array))
print("Class Weights:", class_weights)

# ==== Step 6: Load Pretrained Model ====
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.0005))(x)
x = Dropout(0.5)(x)
predictions = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# ==== Step 7: Compile Model ====
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ==== Step 8: Callbacks ====
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# ==== Step 9: Train Model ====
history = model.fit(
    train,
    validation_data=test,
    epochs=25,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights
)

# ==== Step 10: Save Initial Model ====
model.save("model/brain_tumor_resnet50_stage1.h5")
print("Initial model saved. Starting fine-tuning...")

# ==== Step 10.1: Fine-tune deeper ResNet layers ====
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Recompile with lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Continue training
history_fine = model.fit(
    train,
    validation_data=test,
    epochs=10,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights
)

# Save fine-tuned model
model.save("model/brain_tumor_resnet50_TEST2.h5")
print("Fine-tuned model saved to model/brain_tumor_resnet50_TEST2.h5")


# ==== Step 11: Visualize Accuracy & Loss ====
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='x')
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='x')
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("model/training_performance_resnet.png")
plt.show()

# ==== Step 12: Final Evaluation ====
score = model.evaluate(test)
print("\nFinal Evaluation on Test Data:")
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])
