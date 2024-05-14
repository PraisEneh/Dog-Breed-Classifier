import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, Callback

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

IMG_SIZE = 128
TRAIN_DIR = "train/"
TEST_DIR = "test/"
VALID_DIR = "valid/"
MODEL_PATH = 'models/candidates/'
BASE_MODEL_NAME = 'LR-0.001-CL-6-Drop-0.2-FSize-512-07_28_42.h5'


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
                print(e)
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
            if category in focus_classes or random.random() < 0.3:
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

def create_model(frozen_layers):
    model = load_model(MODEL_PATH + BASE_MODEL_NAME)
    for layer in model.layers[:-frozen_layers]:
        layer.trainable = False
    return model

learning_rates = [1e-5, 1e-4]
rotation_ranges = [30]
zoom_ranges = [0.3]
frozen_layers_options = [5, 8, 10] 
epoch_options = [10, 15, 20]

CATEGORIES = [folder for folder in os.listdir(TRAIN_DIR)]
focus_classes = ['German Sheperd', 'Great Dane', 'Greyhound', 'Malinois', 'Lhasa', 'Shih-Tzu']
X_train, y_train = focused_load_data(TRAIN_DIR, CATEGORIES, focus_classes)
X_test, y_test = focused_load_data(TEST_DIR, CATEGORIES, focus_classes)
X_valid, y_valid = load_data(VALID_DIR, CATEGORIES)


for lr in learning_rates:
    for rotation_range in rotation_ranges:
        for zoom_range in zoom_ranges:
            for frozen_layers in frozen_layers_options:
                for epoch in epoch_options:
                    p = epoch//2
                    model = create_model(frozen_layers)
                    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

                    datagen = ImageDataGenerator(
                        rotation_range=rotation_range,
                        zoom_range=zoom_range,
                        fill_mode='nearest'
                    )

                    name_details = f'EPOCH-{epoch}_LR-{lr}_RR-{rotation_range}_ZR-{zoom_range}_FL-{frozen_layers}'
                    ft_name = f'FROZEN_FT2_{name_details}_{BASE_MODEL_NAME}'
                    checkpoint_path = f'models/finetunes/frozen_finetunes/{ft_name}'
                    callbacks = [
                        TensorBoard(log_dir=f'logs/{ft_name}'),
                        ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
                        EarlyStopping(monitor='val_accuracy', patience=p, restore_best_weights=True)
                    ]

                    history = model.fit(
                        datagen.flow(X_train, y_train, batch_size=32),
                        steps_per_epoch=len(X_train) // 32,
                        epochs=epoch,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks
                    )

                    results = model.evaluate(X_valid, y_valid, batch_size=32)
                    print("valid loss, valid acc:", results)

                    # Generate predictions and analyze them
                    predictions = model.predict(X_valid)
                    predicted_classes = np.argmax(predictions, axis=1)
                    actual_classes = np.argmax(y_valid, axis=1)

                    # Calculate the confusion matrix
                    from sklearn.metrics import confusion_matrix, classification_report
                    cm = confusion_matrix(actual_classes, predicted_classes)
                    print("Confusion Matrix:")
                    print(cm)

                    # Generate a classification report
                    print("Classification Report:")
                    class_report = classification_report(actual_classes, predicted_classes, target_names=CATEGORIES)
                    print(class_report)
                    with open(f'results/{ft_name}', 'a') as f:
                        f.write('Classification Report\n')
                        f.write(class_report)
                        f.write('\n\n')
                        f.write('Confusion Matrix\n')
                        f.write(str(cm))
                        f.write('\n\n')
                        f.write('Validation Results\n')
                        f.write(str(results))
                        f.write('\n')
