import mlflow.sagemaker as mfs

experiment_id = '1'
run_id = '1c00c428833144c58cc14bdda6c71e98'
region = 'us-east-1'
aws_id = '##############'
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