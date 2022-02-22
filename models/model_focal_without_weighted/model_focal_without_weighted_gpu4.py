import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Librairie de path
import pathlib
import pandas as pd

#For model
from focal_loss import SparseCategoricalFocalLoss

#Ligne dessous : verbose initialisation tensorflow
tf.debugging.set_log_device_placement(True)
#get list of gpu
gpus = tf.config.list_physical_devices('GPU')

try:
  # Specify an invalid GPU device
  #tf.config.set_logical_device_configuration(gpus[3],[tf.config.LogicalDeviceConfiguration(memory_limit=64)])
  with tf.device('/device:GPU:3'):

    number_class=1081
    dataset = pathlib.Path(r"/home/data/challenge_2022_miashs/train")
    #train_path = pathlib.Path(r"/home/data/challenge_2022_miashs/train")
    #test_path = pathlib.Path(r"/home/data/challenge_2022_miashs/test")

    #train_path = pathlib.Path(r"C:\Users\lulu5\Documents\train")
    #test_path = pathlib.Path(r"C:\Users\lulu5\Documents\test")

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
        dataset,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        subset = "training",
        target_size=(IMG_H,IMG_W),
        shuffle=True,
        class_mode='binary')

    validation_generator = datagen.flow_from_directory(
        dataset,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        subset = "validation",
        target_size=(IMG_H,IMG_W),
        shuffle=True,
        class_mode='binary')

    #MODELE

    #Modèle instanciation
    model_resnet = tf.keras.applications.resnet50.ResNet50(
        include_top=True, weights=None, input_tensor=None,
        input_shape=None, pooling=None, classes=number_class)

    #Modèle compilation
    model_resnet.compile(optimizer="Adam", 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy', 'sparse_top_k_categorical_accuracy'])
    
    history = model_resnet.fit(train_generator, validation_data=validation_generator,epochs=30,batch_size=BATCH_SIZE, verbose=1,validation_steps=1)

    pd.DataFrame(history.history).to_json("/home/miashs4/results/result_history_resnet.json")

    model_resnet.save("/home/miashs4/results/model_resnet.h5")

except RuntimeError as e:
  print(e)