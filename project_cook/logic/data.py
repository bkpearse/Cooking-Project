"""
Package that contains functions for cleaning and querying our datasets
"""

import os

import pandas as pd
from clean_text import *
from whoosh import index
from whoosh.fields import ID, KEYWORD, TEXT, Schema
from whoosh.qparser import QueryParser

from project_cook.params import *

from clean_text import *
from google.cloud import storage



def load_recipes_from_gcp():

    storage_client = storage.Client(project=GCP_PROJECT)
    blob_name = "RecipeNLG_dataset.csv"

    bucket = storage_client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)

    return pd.read_csv(blob.download_to_file)


def clean_data(df: pd.DataFrame):

    #Keep these just in case
    #data = data.drop('source', axis = 1)
    #data = data.drop('Unnamed: 0', axis = 1)

    #Removing all of the test recipes from df
    df = df[~df['title'].str.contains('Test Recipe')]

    #applying basic cleaning function
    df['NER'] = df['NER'].apply(basic_cleaning)

    #Applying the remove punctuation function
    df['NER'] = df['NER'].apply(remove_punctuation)
    df['ingredients'] = df['ingredients'].apply(remove_punctuation)
    df['directions'] = df['directions'].apply(remove_punctuation)

    #applying the remove word function
    df['NER'] = df['NER'].apply(remove_words)

    return df


def query_data(df: pd.DataFrame, search_term, data_has_header=True):

    # Define the schema of the index
    my_schema = Schema(title=TEXT(stored=True),
                       ingredients=KEYWORD(stored=True, commas=True),
                       directions=TEXT(stored=True),
                       link=ID(stored=True),
                       source=TEXT(stored=True),
                       NER=TEXT(stored=True))

    # Create the index or open it if it already exists, on the cloud
    if not os.path.exists("new_index"):
        os.mkdir("new_index")
        ix = index.create_in("new_index", my_schema)
    else:
        ix = index.open_dir("new_index")
        return ix

    #Index the dataset in chunks
    writer = ix.writer()
    lines = []
    for line in df:  #Is the header a line?
        line = line.strip().split(',')
        if len(line) == 7:
            lines.append(line)
        if len(lines) == CHUNK_SIZE:
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
    print("Indexing complete!")

    search_term = remove_words(search_term)

    # Create a QueryParser for the "NER" field
    qp = QueryParser("NER", schema=ix.schema)

    # Parse the search term
    q = qp.parse(search_term)

    # Search the index and get the results
    results = ix.searcher.search(q)

    recipes = []
    with ix.searcher() as searcher:
        results = searcher.search(q)
        # Print the results
        for result in results:
            # print(result)
            hit = {
                'NER': result['NER'],
                'directions': result['directions'],
                'ingredients': result['ingredients'],
                'link': result['link'],
                'source': result['source'],
                'title': result['title'],
            }
            recipes.append(hit)

    return recipes
