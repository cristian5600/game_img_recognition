print("    Resizing images...")
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import sys
import tensorflow as tf
import glob
from tensorflow import keras
import  os.path
import pathlib
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import cv2
# im_asasini = []
# path =          r"C:\Users\Cristi\source\repos\PythonApplication1\PythonApplication1\assassin"
# img = Image.open(r"C:\Users\Cristi\source\repos\PythonApplication1\PythonApplication1\assassin\imagine.jpg")
# im_asasini.append(img) 
  # This method will show image in any image viewer 
# im_asasini[0].show() 
# image_list = []

# E:\drive\OneDrive\Desktop\games_image_recognition-master\detoate

# image_list = load_images_from_folder(path)

# print(image_list[0])
imagini = []
folder_dir = r"E:\drive\OneDrive\Desktop\games_image_recognition-master\assassin"
# for images in os.listdir(folder_dir):
nr = 0
    # check if the image ends with png
    #if (images.endswith(".jpg")):
     #   print(images)
f = r"E:\drive\OneDrive\Desktop\games_image_recognition-master\detoate\assassin"
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    img = img.resize((300,300))
    img.save(f_img)
    nr=nr+1
f = r"E:\drive\OneDrive\Desktop\games_image_recognition-master\detoate\monopoly"
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    img = img.resize((300,300))
    img.save(f_img)
    nr=nr+1
    print(nr)
f = r"E:\drive\OneDrive\Desktop\games_image_recognition-master\testare"
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    img = img.resize((300,300))
    img.save(f_img)
i = 0



batch_size = nr
img_height = 300
img_width  = 300
f = r"E:\drive\OneDrive\Desktop\games_image_recognition-master\detoate"
train_ds = tf.keras.utils.image_dataset_from_directory(
  f,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(
  f,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
print(train_ds)
print(train_ds.take(1))
import matplotlib.pyplot as plt

class_names = train_ds.class_names
print(class_names)
plt.figure(figsize=(100, 100))
for images, labels in train_ds.take(1):
  #print(labels)
  #print(images)
  print('\n afisare imagini \n')
  for i in range(8):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    # plt.axis("off")
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")
model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

for i in range(5 ):
    f = r"E:\drive\OneDrive\Desktop\games_image_recognition-master\testare"
    for file in os.listdir(f):
        f_img = f+"/"+file
        testarea = f_img
        img = tf.keras.utils.load_img(
            testarea, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        print("This image most likely belongs to {} with a {:.2f} - {}  percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score),file)
    )