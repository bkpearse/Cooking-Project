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



# Create a QueryParser for the "NER" field
qp = QueryParser("NER", schema=ix.schema)

# Parse the search term
q = qp.parse(search_term)


# Search the index and get the results
with ix.searcher() as searcher:
    results = searcher.search(q)

    # Print the results
    for result in results:
        print(result)
