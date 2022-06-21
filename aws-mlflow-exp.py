import mlflow
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from catboost import CatBoostClassifier

mlflow.set_experiment("aws-mlflow-docker")

with mlflow.start_run(run_name="aws-mlflow-experiment") as run:
    param = {
        "objective": "Logloss",
        "colsample_bylevel": 0.050905927922414905,
        "depth": 9,
        "boosting_type": "Ordered",
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 9.951070580524215, }
    mlflow.set_tag("model", "catboost")
    mlflow.set_tag("data scientist", "kb")
    mlflow.log_params(param)
    df = pd.read_csv('heart.csv')
    numerical = df.drop(['HeartDisease'], axis=1).select_dtypes(
        'number').columns
    categorical = df.select_dtypes('object').columns
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    categorical_features_indices = np.where(X.dtypes != np.float)[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    cat_cls = CatBoostClassifier(**param)

    cat_cls.fit(X_train, y_train, eval_set=[
                (X_test, y_test)], cat_features=categorical_features_indices, verbose=0, early_stopping_rounds=100)

    preds = cat_cls.predict(X_test)
    pred_labels = np.rint(preds)
    accuracy = round(accuracy_score(y_test, pred_labels), 4)

    mlflow.log_metric("accuracy", accuracy)

    run_id = run.info.run_uuid
    experiment_id = run.info.experiment_id

    mlflow.sklearn.log_model(cat_cls, artifact_path="models")

    mlflow.end_run()
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")
    print(f"runID: {run_id}")
