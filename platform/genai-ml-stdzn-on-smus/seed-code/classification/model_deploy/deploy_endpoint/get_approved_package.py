# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: MIT-0
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import boto3
import os
from botocore.exceptions import ClientError
from logging import Logger
from config.constants import DEFAULT_DEPLOYMENT_REGION, MODEL_PACKAGE_GROUP_NAME

"""Initialise Logger class"""
logger = Logger(name="deploy_stack")

"""Initialise boto3 SDK resources"""
sm_client = boto3.client("sagemaker", region_name=DEFAULT_DEPLOYMENT_REGION)


def get_approved_package():
    """Gets the latest approved model package for a model package group.
    Priority:
        1. MODEL_PACKAGE_ARN env var (explicit ARN)
        2. Latest Approved package from MODEL_PACKAGE_GROUP_NAME
    Returns:
        The SageMaker Model Package ARN.
    """
    # Priority 1: explicit ARN passed via env var or workflow_dispatch input
    model_package_arn = os.environ.get("MODEL_PACKAGE_ARN")
    if model_package_arn:
        logger.info(f"Using explicitly provided model package ARN: {model_package_arn}")
        return model_package_arn

    # Priority 2: latest approved package from group
    try:
        response = sm_client.list_model_packages(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            MaxResults=100,
        )
        approved_packages = response["ModelPackageSummaryList"]
        while len(approved_packages) == 0 and "NextToken" in response:
            logger.debug(f"Getting more packages for token: {response['NextToken']}")
            response = sm_client.list_model_packages(
                ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
                ModelApprovalStatus="Approved",
                SortBy="CreationTime",
                MaxResults=100,
                NextToken=response["NextToken"],
            )
            approved_packages.extend(response["ModelPackageSummaryList"])
        if len(approved_packages) == 0:
            error_message = f"No approved ModelPackage found for ModelPackageGroup: {MODEL_PACKAGE_GROUP_NAME}"
            logger.error(error_message)
            raise Exception(error_message)
        model_package_arn = approved_packages[0]["ModelPackageArn"]
        logger.info(f"Identified the latest approved model package: {model_package_arn}")
        return model_package_arn
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)
