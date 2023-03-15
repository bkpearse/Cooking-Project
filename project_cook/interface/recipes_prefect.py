import os

import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
from prefect import task, flow

from recipes.interface.main import evaluate, preprocess, train
from recipes.ml_logic.registry import mlflow_transition_model
from recipes.params import *


@task
def preprocess_new_data(min_date: str, max_date: str):
    """
    Run the preprocessing of the new data
    """
    preprocess(min_date=min_date, max_date=max_date)

@task
def evaluate_production_model(min_date: str, max_date: str):
    """
    Run the `Production` stage evaluation on new data
    Returns `eval_mae`
    """
    eval_mae = evaluate(min_date=min_date, max_date=max_date)
    return eval_mae

@task
def re_train(min_date: str, max_date: str):
    """
    Run the training
    Returns train_mae
    """
    train_mae = train(min_date=min_date, max_date=max_date, split_ratio=0.2)
    return train_mae

@task
def notify(old_mae, new_mae):
    """
    Notify about the performance
    """
    base_url = 'https://wagon-chat.herokuapp.com'
    channel = 'krokrob'
    url = f"{base_url}/{channel}/messages"
    author = 'krokrob'
    if new_mae < old_mae and new_mae < 2.5:
        content = f"🚀 New model replacing old in production with MAE: {new_mae} the Old MAE was: {old_mae}"
    elif old_mae < 2.5:
        content = f"✅ Old model still good enough: Old MAE: {old_mae} - New MAE: {new_mae}"
    else:
        content = f"🚨 No model good enough: Old MAE: {old_mae} - New MAE: {new_mae}"
    data = dict(author=author, content=content)
    response = requests.post(url, data=data)
    response.raise_for_status()

@task
def transition_model():
    """
    Transition new model to production
    """
    mlflow_transition_model("Staging", "Production")


@flow(name=PREFECT_FLOW_NAME)
def train_flow():
    """
    build the prefect workflow for the `recipes` package
    """
    min_date = NEW_MONTH_DATE
    max_date = str(datetime.strptime(min_date, "%Y-%m-%d"
                                  ) + relativedelta(months=1)).split()[0]
    preprocessed = preprocess_new_data.submit(min_date=min_date, max_date=max_date)
    old_mae = evaluate_production_model.submit(min_date=min_date, max_date=max_date, wait_for=[preprocessed])
    new_mae = re_train.submit(min_date=min_date, max_date=max_date, wait_for=[preprocessed])
    old_mae = old_mae.result()
    new_mae = new_mae.result()
    if new_mae < old_mae:
        print(f"🚀 New model replacing old in production with MAE: {new_mae} the Old MAE was: {old_mae}")
        transition_model.submit()
    notify.submit(old_mae, new_mae)

if __name__ == "__main__":
    train_flow()
