#Imports
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import scipy
import tensorflow as tf
from keras import layers, models, optimizers
from keras.models import Sequential
from PIL import Image
from tensorflow import keras
from tqdm import tqdm


def proc_img(filepath):
    """
    Create a DataFrame with the filepath and the labels of the pictures
    """

    labels = [str(filepath[i]).split("/")[-2] \
              for i in range(len(filepath))]

    filepath = pd.Series(filepath, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    # Concatenate filepaths and labels
    df = pd.concat([filepath, labels], axis=1)

    # Shuffle the DataFrame and reset index
    df = df.sample(frac=1).reset_index(drop=True)

    return df


def load_image_with_data_augmentation(df):

    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.
        preprocess_input)
    images = generator.flow_from_dataframe(dataframe=df,
                                           x_col='Filepath',
                                           y_col='Label',
                                           target_size=(224, 224),
                                           color_mode='rgb',
                                           class_mode='categorical',
                                           batch_size=32,
                                           shuffle=True,
                                           seed=0,
                                           rotation_range=30,
                                           zoom_range=0.15,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.15,
                                           horizontal_flip=True,
                                           fill_mode="nearest")

    return images


def load_model():

    from keras.applications.vgg16 import VGG16

    model = VGG16(weights="imagenet",
                  include_top=False,
                  input_shape=(224, 224, 3))

    model.trainable = False

    return model


def add_last_layers(model):
    '''Take a pre-trained model and add additional trainable layers on top'''

    flatten_layer = layers.Flatten()
    dense_layer = layers.Dense(500, activation='relu')
    prediction_layer = layers.Dense(36, activation='softmax')

    model = models.Sequential(
        [model, flatten_layer, dense_layer, prediction_layer])

    return model


def build_model():

    model = load_model()
    model = add_last_layers(model)

    opt = optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model
