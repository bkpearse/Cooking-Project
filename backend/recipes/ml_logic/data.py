import pandas as pd
from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    clean raw data by
    - assigning correct dtypes to each colummns
    - removing buggy or irrelevant transactions
    """
    # Compress raw_data by setting types to DTYPES_RAW
    df = df.astype(DTYPES_RAW)

    # remove buggy transactions
    df = df.drop_duplicates()  # TODO: handle in the data source if the data is consumed by chunks
    df = df.dropna(how='any', axis=0)
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0) |
            (df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    df = df[df.passenger_count > 0]
    df = df[df.fare_amount > 0]

    # Remove geographically irrelevant transactions (rows)
    df = df[df.fare_amount < 400]
    df = df[df.passenger_count < 8]
    df = df[df["pickup_latitude"].between(left=40.5, right=40.9)]
    df = df[df["dropoff_latitude"].between(left=40.5, right=40.9)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-73.7)]
    df = df[df["dropoff_longitude"].between(left=-74.3, right=-73.7)]

    print("✅ data cleaned")

    return df

def get_data_with_cache(gcp_project:str,
                        query:str,
                        cache_path:Path,
                        data_has_header=True) -> pd.DataFrame:
    """
    Retrieve `query` data from Big Query, or from `cache_path` if file exists.
    Store at `cache_path` if retrieved from Big Query for future re-use.
    """
    if cache_path.is_file():
        print(Fore.BLUE + "\nLoad data from local CSV..." + Style.RESET_ALL)
        df = pd.read_csv(cache_path, header='infer' if data_has_header else None)

    else:
        print(Fore.BLUE + "\nLoad data from Querying Big Query server..." + Style.RESET_ALL)
        client = bigquery.Client(project=gcp_project)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        # Store as CSV if BQ query returned at least one valid line
        if df.shape[0] > 1:
            df.to_csv(cache_path, header=data_has_header, index=False)

    print(f"✅ Data loaded, with shape {df.shape}")

    return df

def load_data_to_bq(data: pd.DataFrame,
              gcp_project:str,
              bq_dataset:str,
              table: str,
              truncate: bool) -> None:
    """
    - Save dataframe to bigquery
    - Empty the table beforehands if `truncate` is True, append otherwise.
    """
    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(Fore.BLUE + f"\nSave data to bigquery {full_table_name}...:" + Style.RESET_ALL)

    # Load data to full_table_name
    # 🎯 Hint for "*** TypeError: expected bytes, int found":
    # BQ can only accept "str" columns starting with a letter or underscore column

    # TODO: simplify this solution if possible, but student may very well choose another way to do it.
    # We don't test directly against their own BQ table, but only the result of their query.
    data.columns = [f"_{column}" if not str(column)[0].isalpha() and not str(column)[0] == "_"
                                                        else str(column) for column in data.columns]

    client = bigquery.Client()

    # define write mode and schema
    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(f"\n{'Write' if truncate else 'Append'} {full_table_name} ({data.shape[0]} rows)")

    # load data
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()  # wait for the job to complete

    print(f"✅ Data saved to bigquery, with shape {data.shape}")
