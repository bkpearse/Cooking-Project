from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Sequential, load_model
import os
from io import BytesIO
import numpy as np

from colorama import Fore, Style
from PIL import Image
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


def proc_img(filepath):
    """ Create a DataFrame with the filepath and the labels of the pictures
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


def get_model():
    train_dir = Path("notebooks/images/train")
    train_filepaths = list(train_dir.glob(r'**/*.jpg'))

    test_dir = Path("notebooks/images/test")
    test_filepaths = list(test_dir.glob(r'**/*.jpg'))

    val_dir = Path("notebooks/images/validation")
    val_filepaths = list(test_dir.glob(r'**/*.jpg'))
    train_df = proc_img(train_filepaths)
    test_df = proc_img(test_filepaths)
    val_df = proc_img(val_filepaths)
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.
        preprocess_input)

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.
        preprocess_input)

    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
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

    val_images = train_generator.flow_from_dataframe(dataframe=val_df,
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
    test_images = test_generator.flow_from_dataframe(dataframe=test_df,
                                                     x_col='Filepath',
                                                     y_col='Label',
                                                     target_size=(224, 224),
                                                     color_mode='rgb',
                                                     class_mode='categorical',
                                                     batch_size=32,
                                                     shuffle=False)
    model = load_model('notebooks/model.h5', compile=False)
    opt = optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model, train_images


def predict_function(pred_dir):
    model, train_images = get_model()
    pred_filepaths = list(pred_dir.glob(r'**/*.jpg'))
    #Start here in streamlit
    #what is the type of image, then convert it (jpg, jpeg, png, bmp, tiff)
    #pred_df = proc_img(list(user_input))

    pred_df = proc_img(pred_filepaths)
    pred_img_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.
        preprocess_input)

    pred_images = pred_img_generator.flow_from_dataframe(
        dataframe=pred_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False)
    #predict me!
    result = model.predict(pred_images)
    predicted_probabilities = np.argmax(result, axis=1)
    labels = (train_images.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    pred = [labels[k] for k in predicted_probabilities]
    # return {'pred': pred}
    predictions = []
    # zip the predictions with the inputs so we can check if the prediction is correct
    for curr_pred, curr_path in zip(pred, list(pred_df.Filepath)):
        # take the folder name from the file path because the folder name is the type food
        actual = curr_path.split('/')[-2]
        is_correct = curr_pred == actual
        predictions.append({
            'current': curr_pred,
            'actual': actual,
            'is_correct': is_correct
        })
        # print(f'{curr_pred}, {actual}, {is_correct}')
    return predictions
def pred_streamlit(user_input):
    """
    Make a prediction using the latest trained model, using a single image as input
    """
    image = Image.open(BytesIO(user_input))
    print(f"type after Image.open: {type(image)}")

    # Resize to 224 x 224
    image = image.resize((224,224))
    print(f"type after resize: {type(image)}")

    # Convert the image pixels to a numpy array
    image = img_to_array(image)
    print(f"type after img_to_array: {type(image)}")

    # Reshape data for the model
    image = image.reshape((1,224,224,3))
    print(f"type after reshape: {type(image)}")

    # Prepare the image for the VGG model
    image = preprocess_input(image)
    print(f"type after preprocess_input: {type(image)}")

    # Run prediction
    # model = load_model()
    model, train_images = get_model()
    opt = optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    result = model.predict(image)
    predicted_probabilities = np.argmax(result,axis=1)
    # labels = load_labels()
    labels = (train_images.class_indices)
    labels = dict((v, k) for k, v in labels.items())

    # labels = dict(enumerate(labels.flatten()))
    # labels = labels[0]

    prediction = [labels[k] for k in predicted_probabilities]

    return prediction

if __name__ == '__main__':
    print(predict_function(pred_dir=Path("notebooks/images/lemon")))
    # print(os.path.dirname(os.path.realpath(__file__)))
