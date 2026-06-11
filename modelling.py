import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import random
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Eksperimen Wine Quality")


def train_wine_model():
    data = pd.read_csv("wine_preprocessed.csv")
    X = data.drop(columns=['target'])
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        mlflow.autolog()
        model = RandomForestClassifier(n_estimators=100, max_depth=37)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

if __name__ == "__main__":
    train_wine_model()
