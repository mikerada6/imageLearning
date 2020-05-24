import math
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from keras.preprocessing import image

xSize = 185
ySize = 260
channel = 3

print(tf.__version__)

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

batch_size = 20

train_generator = train_datagen.flow_from_directory(
    './tmp/set/Training',  # This is the source directory for training images
    target_size=(xSize, ySize),  # All images will be resized to 150x150
    batch_size=batch_size,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    './tmp/set/Validation',  # This is the source directory for training images
    target_size=(xSize, ySize),  # All images will be resized to 150x150
    batch_size=45,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='categorical')

nb_train_samples = len(train_generator.filenames)
num_classes = len(train_generator.class_indices)

predict_size_train = int(math.ceil(nb_train_samples / batch_size))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                           input_shape=(xSize, ySize, channel)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(validation_generator.num_classes, activation='softmax')
])

# print out what the model looks like
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

modelDesc = 'model_64-32-16-v2'
checkpoint_path = "training/" + modelDesc + ".ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
Path("models").mkdir(parents=True, exist_ok=True)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5,
    steps_per_epoch=93,
    validation_steps=10,
    callbacks=[cp_callback] )

model.save("models/" + modelDesc + ".h5")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

print("done")


folders = [x[0] for x in os.walk("/Users/mradas341/IdeaProjects/imageLearning/tmp/set/Validation")]
labels = ["eld", 'grn', 'iko', 'm20', 'rna', 'thb', 'war']

n = len(labels)
m = len(labels)
confusionMatrix = [[0] * m] * n

count =0
right =0

for i in range(1,len(folders)):
    folder = str(folders[i])
    print(folder)
    files = os.listdir(folders[i])
    for j in range(len(files)):
        file = str(files[j])
        absPath = folder + "/" + file
        img = image.load_img(absPath, target_size=(xSize, ySize))
        # Setting the points for cropped image

        # Cropped image of above dimension
        # (It will not change orginal image)
        im1 = img.crop((100, 0, 240, 185))
        im1 = im1.resize((xSize,ySize))

        x = image.img_to_array(im1)
        x = x / 255
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)

        temp = classes[0]

        print(absPath)
        isPass = str(labels[np.argmax(classes[0])]) in absPath
        if isPass:
            right+=1
        count+=1
        print("\t"+str(labels[np.argmax(classes[0])])+"\t" + str(isPass))
        if j>10:
            break
        confusionMatrix[i][np.argmax(classes[0])]+=1

print(str(right) +"/"+str(count))

print(str(right*100.0/count))
