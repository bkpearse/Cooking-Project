import os

import whoosh.index as index
from whoosh import index
from whoosh.fields import *
from whoosh.fields import KEYWORD, TEXT, Schema
from whoosh.qparser import QueryParser


def settings():
    if os.path.exists("new_index"):
        ix = index.open_dir("new_index")
        return ix
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

    # Create the index or open it if it already exists
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


def search_recipes(search_term):
    ix = settings()
    # Create a QueryParser for the "NER" field
    qp = QueryParser("NER", schema=ix.schema)
    # TODO: Split the string by space.
    q = qp.parse(search_term)

    # Search the index and get the results
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


if __name__ == '__main__':
    print(search_recipes("Fresh chili peppers"))
    # print(os.path.dirname(os.path.realpath(__file__)))
