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