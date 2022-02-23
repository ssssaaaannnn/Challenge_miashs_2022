import os

import warnings
warnings.filterwarnings('ignore', '.*interpolation.*', )

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import mixed_precision

#Librairie de path
import pandas as pd
import numpy as np

#For model
from focal_loss import SparseCategoricalFocalLoss
from tensorflow.keras.callbacks import LearningRateScheduler

################################################
################################################
################################################

NAME_MODEL = "RESNET50_PRETRAINED_FOCAL_WEIGHTED"

# VARIABLE DE CHEMIN
PATH_TRAINSET = "/home/data/challenge_2022_miashs/train"
PATH_TESTSET = "/home/data/challenge_2022_miashs/test"
PATH_STORAGE_MDOEL = "/home/miashs4/results/"+NAME_MODEL+"/" 

# VARIABLE MODEL
EPOCH = 30
BATCH_SIZE = 32
IMG_H = 200
IMG_W = 200
NUM_CLASS = 1081
weights = np.load(r'/home/miashs4/weights_classes.npy')

################################################
################################################
################################################

#Ligne dessous : verbose initialisation tensorflow
tf.debugging.set_log_device_placement(False)
#get list of gpu
gpus = tf.config.list_physical_devices('GPU')


try:
  # Specify an invalid GPU device
  #tf.config.set_logical_device_configuration(gpus[3],[tf.config.LogicalDeviceConfiguration(memory_limit=64)])
  with tf.device('/device:GPU:3'):
    mixed_precision.set_global_policy('mixed_float16')

    #CREATION OF AUGMENTATION
    TFs = {"horizontal_flip": True,
        "vertical_flip": True,
        "rotation_range": 30,
        "featurewise_std_normalization": True,
        "brightness_range": (1, 2)
        }

    datagen = ImageDataGenerator(**TFs)

    #GENERATOR CREATION
    validation_generator = datagen.flow_from_directory(
        PATH_TRAINSET, 
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        subset = "validation",
        target_size=(IMG_H,IMG_W),
        shuffle=True,
        class_mode='binary')

    train_generator = datagen.flow_from_directory(
        PATH_TRAINSET, 
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        subset = "training",
        target_size=(IMG_H,IMG_W),
        shuffle=True,
        class_mode='binary')

    #MODELE

    #Modèle instanciation
    base_model = keras.applications.ResNet50(include_top=False, input_shape=(IMG_H, IMG_W, 3), weights="imagenet")
    base_model.trainable = False

    model = keras.models.Sequential([
            base_model,
            keras.layers.Conv2D(32, (1, 1), activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Flatten(),
            keras.layers.Dense(2048, activation='relu'),
            keras.layers.Dense(NUM_CLASS, activation=None),
            tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')
    ])


    #Fonctionnalité modèle
    earlyStopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=10,
        verbose=0,
        mode="auto",
        baseline=None, # A voir ?
        restore_best_weights=True,
    )

    def step_decay(epoch):
        if epoch < EPOCH/2 + 1:
            return 10**(-4)
        elif epoch < EPOCH*2/3 + 1:
            return 10**(-5)
        else:
            return 10**(-6)
    lrate = LearningRateScheduler(step_decay, verbose=1)

    #Modèle compilation
    model.compile(
        optimizer="Adam", 
        loss=SparseCategoricalFocalLoss(gamma= weights),
        metrics=["top_k_categorical_accuracy"]
    )

    history = model.fit(train_generator, 
        validation_data=validation_generator , epochs=EPOCH,
        batch_size=BATCH_SIZE, verbose=1, validation_steps=1,
        callbacks=[earlyStopping, lrate]
    )

    pd.DataFrame(history.history).to_json(PATH_STORAGE_MDOEL+"result_history_resnet.json")

    model.save(PATH_STORAGE_MDOEL+"model_resnet.h5")

except RuntimeError as e:
  print(e)