import os
import cv2
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, Callback

# Setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

class CyclicalLearningRate(Callback):
    def __init__(self, base_lr=1e-6, max_lr=1e-4, step_size=2000., scale_fn=lambda x: 1.):
        super().__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.scale_fn = scale_fn
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        self.clr_iterations = 0.
        self.model.optimizer.lr = self.base_lr

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.clr_iterations += 1
        self.model.optimizer.lr = self.clr()
        logs['lr'] = self.model.optimizer.lr.numpy()


IMG_SIZE = 128
TRAIN_DIR = "train/"
TEST_DIR = "test/"
CATEGORIES = [folder for folder in os.listdir(TRAIN_DIR)]
training_data = []
test_data = []

# Data loading functions
def load_data(directory, categories):
    data = []
    for category in categories:
        path = os.path.join(directory, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([np.array(new_array), class_num])
            except Exception as e:
                pass
    random.shuffle(data)
    X = np.array([i[0] for i in data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
    y = to_categorical([i[1] for i in data], len(categories))
    return X, y

def focused_load_data(directory, categories, focus_classes):
    data = []
    class_labels = {category: idx for idx, category in enumerate(categories)}

    for category in categories:
        path = os.path.join(directory, category)
        class_num = class_labels[category]
        for img in os.listdir(path):
            if category in focus_classes or random.random() < 0.3:  # Including other classes with less frequency
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    data.append([np.array(new_array), class_num])
                except Exception as e:
                    print(f"Failed to process {img} in {path}: {e}")
    random.shuffle(data)
    X = np.array([i[0] for i in data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
    y = to_categorical([i[1] for i in data], len(categories))
    return X, y

learn_rate=0.0001
rotation_range = 30
zoom_range = 0.4
brightness_range = (0.5, 1.5)

datagen = ImageDataGenerator(
    rotation_range=rotation_range,        # Randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=zoom_range,           # Randomly zoom image 
    fill_mode='nearest'       # Strategy used for filling in newly created pixels
)

PATH = 'models/finetunes/frozen_finetunes/extras/'
NAME = 'FROZEN_FT_RR-30-ZR-0.4-LR-0.0001-LR-0.001-CL-6-Drop-0.2-FSize-512-07_28_42.h5'

FT_NAME='2-'+NAME
checkpoint_path = f'models/finetunes/frozen_finetunes/{FT_NAME}'
LOAD_PATH=PATH+NAME

# # Load the model from the file
model = load_model(LOAD_PATH)

# Freeze all layers except the last few
for layer in model.layers[:-5]:
    layer.trainable = False

# Check the trainable status of each layer
for layer in model.layers:
    print(layer.name, layer.trainable)

# Load data specifically for the underperforming categories
focus_classes = ['German Sheperd', 'Great Dane', 'Greyhound', 'Malinois', 'Lhasa', 'Shih-Tzu']
X_train, y_train = focused_load_data(TRAIN_DIR, CATEGORIES, focus_classes)
X_test, y_test = focused_load_data(TEST_DIR, CATEGORIES, focus_classes)

model.compile(
    optimizer=Adam(learning_rate=learn_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

tb = TensorBoard(log_dir=f'logs/{FT_NAME}')
clr_scheduler = CyclicalLearningRate(base_lr=learn_rate, max_lr=1e-4, step_size=2000)
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# Continue training
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    epochs=10,
    validation_data=(X_test, y_test),
    callbacks=[tb, checkpoint]
)
