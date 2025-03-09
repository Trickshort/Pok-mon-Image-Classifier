import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import MobileNetV2

#  Ensure dataset exists
dataset_path = "/Users/trickshot/Desktop/archive-4/pokemon/pokemon/datasets/pokemon"
train_dir = os.path.join(dataset_path, "training_set")
test_dir = os.path.join(dataset_path, "test_set")

if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    raise Exception(" Training or Test set directories are missing!")

#  Data Augmentation (Improved)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1./255)

#  Load Data
training_set = train_datagen.flow_from_directory(train_dir,
                                                 target_size=(128, 128),
                                                 batch_size=32,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory(test_dir,
                                            target_size=(128, 128),
                                            batch_size=32,
                                            class_mode='categorical')

num_classes = len(training_set.class_indices)

#  Save class mappings
with open("pokemon_mapping.json", "w") as f:
    json.dump(training_set.class_indices, f)

print(f" Loaded {num_classes} Pokémon classes!")

#  Check class balance
print("\n Checking dataset balance:")
for class_name, index in training_set.class_indices.items():
    class_count = len(os.listdir(os.path.join(train_dir, class_name)))
    print(f" {class_name}: {class_count} images")

#  Build CNN Model (Deep + Regularization)
cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=[128, 128, 3], kernel_regularizer=l2(0.001)))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001)))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001)))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001)))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=512, activation='relu', kernel_regularizer=l2(0.001)))
cnn.add(tf.keras.layers.Dropout(0.5))

cnn.add(tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=l2(0.001)))
cnn.add(tf.keras.layers.Dropout(0.5))

cnn.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

#  Compile Model with Lower Learning Rate
cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

#  ModelCheckpoint to Save Best Model
checkpoint = ModelCheckpoint("pokemon_cnn_best.h5", monitor='val_accuracy', save_best_only=True, mode='max')

#  Train the Model (100 Epochs)
trained_model = cnn.fit(x=training_set,
                         validation_data=test_set,
                         epochs=100,
                         callbacks=[checkpoint])

# Save Final Model
cnn.save("pokemon_cnn_final.h5")
print("Model saved as `pokemon_cnn_final.h5`")

#  Plot Training Accuracy
plt.figure(figsize=(8,5))
plt.plot(trained_model.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(trained_model.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Training & Validation Accuracy")
plt.grid()
plt.show()
