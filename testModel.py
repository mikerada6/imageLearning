import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import numpy as np


def getMax(array):
    max = -1000
    bestIndex = 0
    for i in range(len(array)):
        if array[i] > max:
            max = array[i]
            bestIndex = i
    return max, bestIndex


print(tf.version.VERSION)

xSize = 185
ySize = 260
channel = 3

modelDesc = 'model_64-32-16-v2'
modelLocation = "models/" + modelDesc + ".h5"

model = tf.keras.models.load_model(modelLocation)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = test_datagen.flow_from_directory(
    './tmp/set/Training',  # the source directory for training images
    target_size=(xSize, ySize),  # All images will be resized to 150x150
    batch_size=577,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='categorical')

valid_generator = test_datagen.flow_from_directory(
    './tmp/set/Validation',  # the source directory for training images
    target_size=(xSize, ySize),  # All images will be resized to 150x150
    batch_size=1,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    './tmp/captured',  # the source directory for training images
    target_size=(xSize, ySize),  # All images will be resized to 150x150
    batch_size=1,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='categorical')

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

print("start")
probabilities = model.evaluate(test_generator, STEP_SIZE_VALID)
print(probabilities)
print("end")

print("start")
probabilities = model.predict(test_generator, STEP_SIZE_VALID)
print("end")

y_true = np.array([0] * 1000 + [1] * 1000)
y_pred = probabilities > 0.5

matrix = confusion_matrix(y_true, y_pred)

print("DONE")
