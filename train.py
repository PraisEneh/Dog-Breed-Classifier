import os
import cv2
import math
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

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

# Load data
X_train, y_train = load_data(TRAIN_DIR, CATEGORIES)
X_test, y_test = load_data(TEST_DIR, CATEGORIES)

# Hyperparameters grid
learning_rates = [0.001]
conv_layers = [6]
dropouts = [0.2]
filter_sizes = [512]

# Grid search
best_accuracy = 0
best_params = {}
best_model_path = ""


# Define data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,        # Randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,    # Randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,   # Randomly shift images vertically (fraction of total height)
    shear_range=0.2,          # Shear Intensity (Shear angle in counter-clockwise direction in degrees)
    zoom_range=0.2,           # Randomly zoom image 
    horizontal_flip=True,     # Randomly flip images
    fill_mode='nearest'       # Strategy used for filling in newly created pixels
)

class CyclicalLearningRate(Callback):
    def __init__(self, base_lr=1e-4, max_lr=1e-2, step_size=2000., scale_fn=lambda x: 1.):
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

def create_resnet_like_model(conv_layer, dropout, filter_size):
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = Conv2D(filter_size, (3, 3), padding='same', activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)

    # Initial convolution and pooling
    for _ in range(conv_layer - 1):
        identity = x
        x = Conv2D(filter_size, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filter_size, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x, identity])  # Adding skip connection
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2))(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(filter_size, activation='relu')(x)
    x = Dropout(dropout)(x)
    outputs = Dense(len(CATEGORIES), activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def create_model(conv_layer, dropout, filter_size):
    model = Sequential([
    layers.Conv2D(filter_size, (3, 3), padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2))
    ])

    for _ in range(conv_layer - 1):
        model.add(layers.Conv2D(filter_size, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(filter_size, activation='relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(len(CATEGORIES), activation='softmax'))
    return model

def create_advanced_model(conv_layer, dropout, filter_size):
    model = Sequential()
    model.add(Conv2D(filter_size, (3, 3), padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(MaxPooling2D((2, 2)))

    for _ in range(conv_layer - 1):
        model.add(Conv2D(filter_size, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(filter_size, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(filter_size // 2, activation='relu'))  # Reducing the size of subsequent layers
    model.add(Dropout(dropout * 0.5))
    model.add(Dense(len(CATEGORIES), activation='softmax'))
    return model


for learn_rate in learning_rates:
    for conv_layer in conv_layers:
        for dropout in dropouts:
            for filter_size in filter_sizes:
                K.clear_session()  # Clear previous models from memory to prevent resource clutter
                NAME = f'LR-{learn_rate}-CL-{conv_layer}-Drop-{dropout}-FSize-{filter_size}-{datetime.now().strftime("%H_%M_%S")}'
                print(NAME)
                checkpoint_path = f'models/{NAME}.h5'
                tb = TensorBoard(log_dir=f'logs/{NAME}')

                checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
                early_stop = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)
                lr_scheduler = CyclicalLearningRate(base_lr=1e-4, max_lr=0.002, step_size=2000.)


                model = create_resnet_like_model(conv_layer, dropout, filter_size)
                model.compile(optimizer=Adam(learning_rate=learn_rate), loss=categorical_crossentropy, metrics=['accuracy'])
                history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                            steps_per_epoch=len(X_train) // 32,
                            epochs=100,
                            validation_data=(X_test, y_test),
                            callbacks=[tb, checkpoint, lr_scheduler],
                            verbose=2)

                final_accuracy = max(history.history['val_accuracy'])
                if final_accuracy > best_accuracy:
                    best_accuracy = final_accuracy
                    best_params = {'learning_rate': learn_rate, 'conv_layers': conv_layer, 'dropout': dropout, 'filter size': filter_size}
                    best_model_path = checkpoint_path


print("Best Accuracy:", best_accuracy)
print("Best Parameters:", best_params)
with open('best_params.txt', 'a') as f:
    f.write(str(best_params)+ ' Accuracy: '+ str(best_accuracy) +'\n')
