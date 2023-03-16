import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from whoosh.fields import *
from whoosh.index import create_in, open_dir
from whoosh.qparser import QueryParser
import whoosh.index as index
from whoosh.fields import Schema, TEXT
from whoosh.qparser import MultifieldParser
from whoosh.qparser import OrGroup
from whoosh.fields import Schema, KEYWORD
from whoosh.query import Phrase
from whoosh import qparser
from whoosh.query import Or
from whoosh.query import Term, And, BooleanQuery
from whoosh.index import open_dir
from whoosh import scoring
from whoosh import index
import os


def settings():

    # # SETTINGS
    # from google.colab import drive

    # drive.mount('/content/drive')

    # import os
    # os.chdir('/content/drive/MyDrive/Colab Notebooks/')
    filename = 'project_cook/data/full_dataset.csv'
    df = pd.read_csv(filename)

    # Define the schema of the index
    my_schema = Schema(title=TEXT(stored=True),
                    ingredients=KEYWORD(stored=True, commas=True),
                    directions=TEXT(stored=True),
                    link=ID(stored=True),
                    source=TEXT(stored=True),
                    NER=TEXT(stored=True))

    # Create the index or open it if it already exists
    if not os.path.exists("new_index"):
        os.mkdir("new_index")
        ix = index.create_in("new_index", my_schema)
    else:
        ix = index.open_dir("new_index")
        return ix

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

def search_recipes(search_term):
    ix = settings()
    # Create a QueryParser for the "NER" field
    qp = QueryParser("NER", schema=ix.schema)
    # Parse the search term
    q = qp.parse(search_term)

    # Search the index and get the results
    recipes = []
    with ix.searcher() as searcher:
        results = searcher.search(q)
        # Print the results
        for result in results:
            # print(result)
            hit = {
                'NER':result['NER'],
                'directions': result['directions'],
                'ingredients': result['ingredients'],
                'link': result['link'],
                'source': result['source'],
                'title': result['title'],
            }
            recipes.append(hit)
    return recipes


if __name__ == '__main__':
    print(search_recipes("Fresh chili peppers"))
    # print(os.path.dirname(os.path.realpath(__file__)))
