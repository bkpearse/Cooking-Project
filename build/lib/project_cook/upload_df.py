import os

import numpy as np
import pandas as pd
from google.cloud import bigquery

PROJECT = os.environ.get("GCP_PROJECT")
DATASET = os.environ.get("BQ_DATASET")
TABLE = "recipes"

table = f"{PROJECT}.{DATASET}.{TABLE}"

original_df = pd.read_csv(
    'recipes/data/full_dataset.csv',
    names=['title', 'ingredients', 'directions', 'link', 'source', 'NER'])
client = bigquery.Client()

df = original_df[1100000:1500000]
print(df.shape)
# df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

# df_split = np.array_split(df, 1000)
# for df_chunk in df_split:
write_mode = "WRITE_APPEND"  #"WRITE_TRUNCATE" # or
job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

job = client.load_table_from_dataframe(df, table, job_config=job_config)
result = job.result()
