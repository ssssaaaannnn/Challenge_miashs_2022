import tensorflow as tf

import pandas as pd
import numpy as np
from glob import glob
import os

#test_dir = r"D:/data/challenge_2022_miashs/test"
#model = tf.keras.models.load_model(r"C:\Users\jchik\Documents\m2\challenge\modele\model_V1\model_resnet.h5")
model = tf.keras.models.load_model("/home/miashs4/results/resnet_models")
test_dir = "/home/data/challenge_2022_miashs/test"

with tf.device('/GPU:0'):

    test_datagen  = tf.keras.preprocessing.image.ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        color_mode="rgb",
        shuffle = False,
        class_mode='categorical',
        batch_size=1
    )
    filenames = test_generator.filenames
    nb_samples = len(filenames)
    predict = model.predict_generator(test_generator,steps = nb_samples)
    
    labels = [os.path.basename(lab) for lab in glob("D:/data/challenge_2022_miashs/train/*")]
    
label_predit = []
for pred in predict:
    idx = np.argmax(pred)
    label_predit.append(labels[idx])
    
df = pd.DataFrame({"Id": filenames, "Category": label_predit})
df.Id = df.Id.map(lambda x: x[2:-4])
df.to_csv("prediction_v1.csv", sep = ",", index=False)