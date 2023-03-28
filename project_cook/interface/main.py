import os
import pickle
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import whoosh.index as index
from google.cloud import storage
from googletrans import LANGUAGES, Translator
from keras.applications.vgg16 import preprocess_input
from PIL import Image
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from whoosh import index
from whoosh.fields import *
from whoosh.fields import KEYWORD, TEXT, Schema
from whoosh.qparser import QueryParser

from project_cook.logic.clean_text import *

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
    """
    Returns the model and the train_images for labels.
    """
    train_dir = Path("notebooks/images/train")
    train_filepaths = list(train_dir.glob(r'**/*.jpg'))
    train_df = proc_img(train_filepaths)
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
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

    model = load_model('notebooks/model.h5', compile=False)
    opt = optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model, train_images


def pred_streamlit(user_input, lang):
    """
    Make a prediction using the latest trained model, using a single image as input
    """
    image = Image.open(BytesIO(user_input))

    # Resize to 224 x 224
    image = image.resize((224, 224))

    # Convert the image pixels to a numpy array
    image = img_to_array(image)

    # Reshape data for the model
    image = image.reshape((1, 224, 224, 3))

    # Prepare the image for the VGG model
    image = preprocess_input(image)
    print(f"type after preprocess_input: {type(image)}")

    # Run prediction
    model, train_images = get_model()
    opt = optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    result = model.predict(image)
    predicted_probabilities = np.argmax(result, axis=1)
    labels = (train_images.class_indices)
    labels = dict((v, k) for k, v in labels.items())

    prediction = [labels[k] for k in predicted_probabilities]
    if lang == 'en':
        transl = prediction
    else:
        transl = [
            translate_text(labels[k], lang) for k in predicted_probabilities
        ]
    return {'prediction': prediction, 'translation': transl}


def settings(filename='project_cook/data/full_dataset.csv'):
    """
    Sets the index for recipes search.
    """
    if os.path.exists("new_index"):
        ix = index.open_dir("new_index")
        return ix

    # Define the schema of the index
    my_schema = Schema(title=TEXT(stored=True),
                       ingredients=KEYWORD(stored=True, commas=True),
                       directions=TEXT(stored=True),
                       link=ID(stored=True),
                       source=TEXT(stored=True),
                       NER=TEXT(stored=True))

    # # Create the index or open it if it already exists
    os.mkdir("new_index")
    ix = index.create_in("new_index", my_schema)

    # Set the chunk size
    chunk_size = 10000

    # Index the dataset in chunks
    writer = ix.writer()
    with open(filename) as f:
        next(f)  # Skip the header row
        lines = []
        for line in f:
            line = line.strip().split(',')
            if len(line) == 7:
                lines.append(line)
            if len(lines) == chunk_size:
                for l in lines:
                    writer.add_document(title=l[1],
                                        ingredients=l[2],
                                        directions=l[3],
                                        link=l[4],
                                        source=l[5],
                                        NER=l[6])
                lines = []
                writer.commit()
                writer = ix.writer()
        # Add any remaining lines
        for l in lines:
            writer.add_document(title=l[1],
                                ingredients=l[2],
                                directions=l[3],
                                link=l[4],
                                source=l[5],
                                NER=l[6])
        writer.commit()

    return ix


def search_recipes(search_term, lang):
    """
    Search recipes.

    Args:
        search_term (_type_): the search term(s), e,g. 'apple pear cinnamon'
        lang (_type_): language code, e.g. 'en'
    """

    # Clean the terms.
    search_term = basic_cleaning(search_term)
    search_term = remove_punctuation(search_term)
    search_term = remove_words(search_term)

    ix = settings()
    # Create a QueryParser for the "NER" field
    qp = QueryParser("NER", schema=ix.schema)
    q = qp.parse(search_term)

    # Search the index and get the results
    recipes = []
    with ix.searcher() as searcher:
        results = searcher.search(q)
        # Print the results
        for result in results:
            if lang == 'en':
                hit = {
                    'NER': result['NER'],
                    'directions': result['directions'],
                    'ingredients': result['ingredients'],
                    'link': result['link'],
                    'source': result['source'],
                    'title': result['title'],
                }
            else:
                hit = {
                    'NER': translate_text(result['NER'], lang),
                    'directions': translate_text(result['directions'], lang),
                    'ingredients': translate_text(result['ingredients'], lang),
                    'link': result['link'],
                    'source': result['source'],
                    'title': translate_text(result['title'], lang),
                }
            recipes.append(hit)
    return recipes


def translate_text(text, target_language="en"):
    """
    Translation function.
    """
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text


if __name__ == '__main__':
    try:
        settings()
    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
