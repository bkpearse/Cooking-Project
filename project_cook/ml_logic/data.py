"""
Package that contains functions for cleaning and querying our datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from whoosh import index
from whoosh.fields import Schema, TEXT, KEYWORD, ID
from whoosh.qparses import QueryParser

import os
from project_cook.params import *


def load_data():
    pass

def query_data(df: pd.DataFrame,
               search_term: str,
               data_has_header = True):

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

    #Index the dataset in chunks
    writer = ix.writer()
    lines = []
    for line in df: #Is the header a line?
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

    # Create a QueryParser for the "NER" field
    qp = QueryParser("NER", schema=ix.schema)

    # Parse the search term
    q = qp.parse(search_term)

    # Search the index and get the results
    results = ix.searcher.search(q)

    #can we get the term from results now?

    pass

def clean_data():
    pass
