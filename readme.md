# MLFLOW - AWS SAGEMAKER
Mlflow- AWS Sagemaker folder for Data Scientist Trainees and DS Aspirants.


![MLflow-models](https://user-images.githubusercontent.com/51021282/174871238-7350487d-a4ac-42db-afd4-3c5232f01310.jpeg)


New environment

```bash
    conda create -n mlops-aws-docker python=3.9 -y
```

Activate the environment

```bash
    conda activate mlops-aws-docker
```

Requirements File

```bash
    touch requirements.txt
```

Connect to Github

```bash
git init
```

```bash
git add .
```


```bash
git commit -m "first commit"
```

oneliner updates for readme

```bash
git add . && git commit -m "update Readme.md"
```

pushing changes to new repo

`````bash
git remote add origin git@github.com:kb1907/mlflow-aws-sagemaker.git
git branch -M main
git push -u origin main

Setup AWS

Prepare MlFlow Experiment

```` py
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
`````

- Open the <i>Mlflow</i> user interface in <i>http://127.0.0.1:5000</i>.

<h2>Deploy the model to AWS</h2>
<p><h3>Build a Docker Image and push it to AWS ECR</h3>

![Docker-Image](https://user-images.githubusercontent.com/51021282/174871564-6351c034-f469-4167-b4ef-8fece5cc9c3e.png)


![AWS-ECR-Image](https://user-images.githubusercontent.com/51021282/174871642-022f68bf-2d83-4dc4-98a9-6b1eb80d73b5.png)


<p><h3>Deploy image to Sagemaker</h3>

![sagemaker-endpoint](https://user-images.githubusercontent.com/51021282/174871702-684b88f1-e609-4407-be48-875eee4ba52b.jpeg)

```py
import mlflow.sagemaker as mfs

experiment_id = '1'
run_id = '1c00c428833144c58cc14bdda6c71e98'
region = 'us-east-1'
aws_id = '#############'
arn = 'arn:aws:iam::###########:role/sagemaker-for-deploy-ml-model'
app_name = 'model-application'
model_uri = f'mlruns/{experiment_id}/{run_id}/artifacts/models'
tag_id = '1.26.1'

image_url = aws_id + '.dkr.ecr.' + region + '.amazonaws.com/mlflow-pyfunc:' + tag_id

mfs.deploy(app_name=app_name,
           model_uri=model_uri,
           region_name=region,
           mode="create",
           execution_role_arn=arn,
           image_url=image_url)

```

![sagemaker-endpoint-aws](https://user-images.githubusercontent.com/51021282/174871794-d99b571f-090c-4ccf-8bb6-b7a35e4f8367.png)



<h2>Use the model with the new data</h2>

```py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import json
import boto3

global app_name
global region
app_name = 'model-application'
region = 'us-east-1'

def check_status(app_name):
sage_client = boto3.client('sagemaker', region_name=region)
endpoint_description = sage_client.describe_endpoint(EndpointName=app_name)
endpoint_status = endpoint_description["EndpointStatus"]
return endpoint_status

def query_endpoint(app_name, input_json):
client = boto3.session.Session().client("sagemaker-runtime", region)

    response = client.invoke_endpoint(
        EndpointName=app_name,
        Body=input_json,
        ContentType='application/json; format=pandas-split',
    )
    preds = response['Body'].read().decode("ascii")
    preds = json.loads(preds)
    print("Received response: {}".format(preds))
    return preds

## check endpoint status

print("Application status is: {}".format(check_status(app_name)))

# Prepare data to give for predictions

df = pd.read_csv('heart.csv')
X= df.drop('HeartDisease', axis=1)
y= df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

## create test data and make inference from enpoint

query_input = pd.DataFrame(X_test).iloc[[15]].to_json(orient="split")
prediction = query_endpoint(app_name=app_name, input_json=query_input)
```
