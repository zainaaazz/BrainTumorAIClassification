# === Brain Tumor Classification using DenseNet121 (Small Dataset Version) ===

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# ==== Step 1: Paths ====
train_path = "data/Training"
test_path = "data/Testing"

# ==== Step 2: Count Images ====
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
plt.title('Training Data')

plt.subplot(1, 2, 2)
sns.barplot(x='Tumor_Type', y='Image_Count', data=test_df)
plt.title('Testing Data')
plt.tight_layout()
os.makedirs("model", exist_ok=True)
plt.savefig("model/class_distribution_small.png")
plt.close()

# ==== Step 4: Data Preprocessing ====
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train = train_datagen.flow_from_directory(
    train_path, target_size=(224, 224), batch_size=16, class_mode='categorical'
)
test = test_datagen.flow_from_directory(
    test_path, target_size=(224, 224), batch_size=16, class_mode='categorical'
)

# ==== Step 5: Class Weights ====
class_labels = list(train.class_indices.keys())
total_train = sum(train_df['Image_Count'])
class_counts = train_df.set_index('Tumor_Type').loc[class_labels]['Image_Count'].to_numpy()
class_weights_array = compute_class_weight(class_weight='balanced',
                                           classes=np.arange(len(class_labels)),
                                           y=np.repeat(np.arange(len(class_labels)), class_counts))
class_weights = dict(enumerate(class_weights_array))
print("Class Weights:", class_weights)

# ==== Step 6: Load Pretrained DenseNet121 ====
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze all but top 20 layers
for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.5)(x)
output = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# ==== Step 7: Compile ====
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ==== Step 8: Callbacks ====
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
checkpoint = ModelCheckpoint("model/best_model_densenet_small.h5", monitor="val_accuracy",
                             save_best_only=True, verbose=1)

# ==== Step 9: Train ====
history = model.fit(
    train,
    validation_data=test,
    epochs=25,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# ==== Step 10: Save Final ====
model.save("model/final_densenet_small.keras")
print("Model saved to model/final_densenet_small.keras")

# ==== Step 11: Plot ====
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("model/training_plot_densenet_small.png")
plt.close()

# ==== Step 12: Evaluate ====
score = model.evaluate(test)
print("\nFinal Evaluation on Test Data:")
print(f"Test Loss: {score[0]:.4f}")
print(f"Test Accuracy: {score[1]*100:.2f}%")
