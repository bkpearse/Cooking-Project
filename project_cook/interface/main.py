import os
import pickle
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import whoosh.index as index
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
from googletrans import Translator, LANGUAGES

from google.cloud import storage
from whoosh import store

class GoogleCloudStorage(store.Storage):
    def __init__(self, bucket_name, client=None):
        self.bucket_name = bucket_name
        if client is None:
            client = storage.Client()
        self.client = client
        self.bucket = self.client.get_bucket(self.bucket_name)

    def create(self, name):
        return store.RAMFile(name, self)

    def open(self, name, *args, **kwargs):
        blob = self.bucket.get_blob(name)
        if blob is None:
            raise store.NoSuchFileError(f"No file found with the name: {name}")
        return store.RAMFile(blob.download_as_bytes(), self)

    def list(self):
        return [blob.name for blob in self.bucket.list_blobs()]

    def remove(self, name):
        blob = self.bucket.get_blob(name)
        if blob is not None:
            blob.delete()

    def rename(self, src, dest):
        src_blob = self.bucket.get_blob(src)
        if src_blob is None:
            raise store.NoSuchFileError(f"No file found with the name: {src}")

        self.bucket.copy_blob(src_blob, self.bucket, new_name=dest)
        src_blob.delete()

    def exists(self, name):
        return self.bucket.get_blob(name) is not None

    def _path(self, name):
        return f"gs://{self.bucket_name}/{name}"



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


def get_model_pickle():

    pickled_model = pickle.load(open('notebooks/model_weights.pkl', 'rb'))

    print(pickled_model)
    # pickled_model.predict(X_test)


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
    # model = load_model()
    model, train_images = get_model()
    opt = optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    result = model.predict(image)
    predicted_probabilities = np.argmax(result, axis=1)
    # labels = load_labels()
    labels = (train_images.class_indices)
    labels = dict((v, k) for k, v in labels.items())

    # labels = dict(enumerate(labels.flatten()))
    # labels = labels[0]

    transl = [translate_text(labels[k], lang) for k in predicted_probabilities]
    prediction = [labels[k] for k in predicted_probabilities]

    return {'prediction': prediction, 'translation': transl}


def settings():

    bucket_name = "whatcanicook_v1nc3nz00"  # Replace with your actual bucket name
    storage_obj = GoogleCloudStorage(bucket_name)

    try:
        ix = index.open_dir(storage_obj)
    except index.EmptyIndexError:
        ix = None

    # Saving Index locally
    #if os.path.exists("new_index"):
        #ix = index.open_dir("new_index")
        #return ix
    # # SETTINGS
    # from google.colab import drive

    # drive.mount('/content/drive')

    # import os
    # os.chdir('/content/drive/MyDrive/Colab Notebooks/')
    filename = 'project_cook/data/full_dataset.csv'

    # Define the schema of the index
    my_schema = Schema(title=TEXT(stored=True),
                       ingredients=KEYWORD(stored=True, commas=True),
                       directions=TEXT(stored=True),
                       link=ID(stored=True),
                       source=TEXT(stored=True),
                       NER=TEXT(stored=True))

    ix = index.create_in(storage_obj, my_schema)

    # Create the index or open it if it already exists
    #os.mkdir("new_index")
    #ix = index.create_in("new_index", my_schema)

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
    ix = settings()
    # Create a QueryParser for the "NER" field
    qp = QueryParser("NER", schema=ix.schema)
    # TODO: Split the string by space.
    search_term = basic_cleaning(search_term)
    search_term = remove_punctuation(search_term)
    search_term = remove_words(search_term)
    q = qp.parse(search_term)

    # Search the index and get the results
    recipes = []
    with ix.searcher() as searcher:
        results = searcher.search(q)
        # Print the results
        for result in results:
            # print(result)
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




# if __name__ == '__main__':
#     try:
#         preprocess_and_train()
#         # preprocess()
#         # train()
#         pred()
#     except:
#         import sys
#         import traceback

#         import ipdb
#         extype, value, tb = sys.exc_info()
#         traceback.print_exc()
#         ipdb.post_mortem(tb)


def translate_text(text, target_language="en"):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text
