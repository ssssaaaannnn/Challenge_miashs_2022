# Librairie IA
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Librairie de path
import pathlib
import pandas as pd

#For model
from focal_loss import SparseCategoricalFocalLoss

INITIAL_PATH = r"C:\Users\lulu5\Documents"

train_path = pathlib.Path(INITIAL_PATH+"\train")
test_path = pathlib.Path(INITIAL_PATH+"\test")

BATCH_SIZE = 32
IMG_H = 200
IMG_W = 200

#CREATION OF AUGMENTATION
TFs = {"horizontal_flip": True,
       "vertical_flip": True,
       "rotation_range": 30,
       "featurewise_std_normalization": True,
       "brightness_range": (1, 2)
       }

datagen = ImageDataGenerator(**TFs)

#GENERATOR CREATION
train_generator = datagen.flow_from_directory(
    train_path,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    target_size=(IMG_H, IMG_W),
    shuffle=True,
    class_mode='categorical')

test_flow = datagen.flow_from_directory(
    test_path,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    target_size=(IMG_H, IMG_W),
    shuffle=True,
    class_mode='categorical')

#MODELE

#Modèle instanciation
model = tf.keras.applications.resnet50.ResNet50(
    include_top=True, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None, classes=1000)

#Modèle compilation
history = model.compile(optimizer="Adam", 
        loss = SparseCategoricalFocalLoss(gamma=5),
        metrics=['accuracy', 'sparse_top_k_categorical_accuracy'])

pd.DataFrame(history.history).to_json("result_history_v1.json")

model.save("model_V1.h5")