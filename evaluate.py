import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

IMG_SIZE = 128
TEST_DIR = "valid/"
CATEGORIES = [folder for folder in os.listdir(TEST_DIR)]

# Data loading function
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

# Load the data
X_test, y_test = load_data(TEST_DIR, CATEGORIES)

# Load the model from the file
PATH = 'models/finetunes/frozen_finetunes/'
NAME = '2-FROZEN_FT_RR-30-ZR-0.4-LR-0.0001-LR-0.001-CL-6-Drop-0.2-FSize-512-07_28_42.h5'
LOAD_PATH = PATH + NAME
model = load_model(LOAD_PATH)

# Evaluate the model on the test set
results = model.evaluate(X_test, y_test, batch_size=32)
print("test loss, test acc:", results)

# Optionally, generate predictions and analyze them
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = np.argmax(y_test, axis=1)

# Calculate the confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(actual_classes, predicted_classes)
print("Confusion Matrix:")
print(cm)

# Generate a classification report
print("Classification Report:")
print(classification_report(actual_classes, predicted_classes, target_names=CATEGORIES))
