# Lab 1 – Onboarding

**Persona:** Platform Admin

Set up the ML platform infrastructure and onboard users onto it. This lab establishes the foundation the rest of the workshop builds on, using **Amazon SageMaker AI (SMAI)**, including governed access to datasets in Amazon S3.

## What you'll do
- Provision the **SMAI** platform infrastructure (SageMaker domain, VPC, and supporting resources).
- Onboard users and configure their access.
- Establish governed access to data in S3.

## Contents
- `user-data-governance.ipynb` – Read data from S3 and explore governed data access patterns.

## Related assets
- `../templates/` – CloudFormation templates for the SageMaker domain, VPC, and MLOps toolchain.
- `../scripts/delete_sagemaker_domain.py` – Tear down the SageMaker domain during clean-up.
