"""Sets ModelLifeCycle on the latest model package in a group."""
import boto3
import os
import sys

region = os.environ["REGION"]
group_name = os.environ["MODEL_PACKAGE_GROUP_NAME"]
pipeline_name = os.environ["PIPELINE_NAME"]

client = boto3.client("sagemaker", region_name=region)

# Get latest model package
resp = client.list_model_packages(
    ModelPackageGroupName=group_name,
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=1,
)

packages = resp.get("ModelPackageSummaryList", [])
if not packages:
    print(f"No model packages found in group {group_name}, skipping.")
    sys.exit(0)

arn = packages[0]["ModelPackageArn"]
print(f"Found model package: {arn}")

# Get the latest pipeline execution to find the training job ARN
executions = client.list_pipeline_executions(
    PipelineName=pipeline_name,
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=1,
)
execution_arn = executions["PipelineExecutionSummaries"][0]["PipelineExecutionArn"]

steps = client.list_pipeline_execution_steps(PipelineExecutionArn=execution_arn)
training_job_arn = None
for step in steps["PipelineExecutionSteps"]:
    if "TrainingJob" in step.get("Metadata", {}):
        training_job_arn = step["Metadata"]["TrainingJob"]["Arn"]
        break

print(f"Setting lifecycle on: {arn}")
update_kwargs = {
    "ModelPackageArn": arn,
    "ModelLifeCycle": {
        "Stage": "development",
        "StageStatus": "InProgress",
        "StageDescription": "Model registered via GitHub Actions pipeline, pending review",
    },
}

if training_job_arn:
    print(f"Linking training job: {training_job_arn}")
    update_kwargs["SourceUri"] = training_job_arn

client.update_model_package(**update_kwargs)
print(f"Model package updated successfully for {arn}")
