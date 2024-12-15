import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# Proje kök dizini
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
train_dir = os.path.join(BASE_DIR, 'dataset', 'train')
test_dir = os.path.join(BASE_DIR, 'dataset', 'test')
saved_model_dir = os.path.join(BASE_DIR, 'saved_model')
os.makedirs(saved_model_dir, exist_ok=True)

# Veri işleme
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=32, class_mode='categorical')
test_data = test_datagen.flow_from_directory(
    test_dir, target_size=(150, 150), batch_size=32, class_mode='categorical')

# Model tanımlama
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model eğitimi
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10,
    steps_per_epoch=train_data.samples // train_data.batch_size,
    validation_steps=test_data.samples // test_data.batch_size
)

# Modeli kaydet
model.save(os.path.join(saved_model_dir, 'bird_classifier.h5'))
print("Model başarıyla kaydedildi.")
