---
inclusion: always
---

# AWS Profile & Region

When running any AWS CLI command for this workspace, always use the **`tfc`** named
profile and the **`us-east-1`** region unless the user explicitly says otherwise.

- Append `--profile tfc --region us-east-1` to every `aws ...` invocation (adjust
  the region to match the region you deployed the workshop stack in).
- The workshop stack (`bank-marketing-prediction-workshop`) and all its resources
  live in your workshop account. Confirm the active account/region before running
  CLI commands: `aws sts get-caller-identity` and
  `aws cloudformation describe-stacks --stack-name bank-marketing-prediction-workshop`.
- The SageMaker Studio domain, its data buckets, the Glue/Athena database
  `bank_marketing`, and the inference-capture SQS queue all live in the workshop
  account/region — a resource ARN showing a different region is a signal something
  is misconfigured, not a normal variation.
- Prefer read-only operations; confirm before any destructive or production-affecting change (see production-safety rules).

```bash
# Correct
aws sts get-caller-identity --profile tfc --region us-east-1
aws cloudformation describe-stacks --stack-name bank-marketing-prediction-workshop --profile tfc --region us-east-1

# Wrong — missing profile/region (will use default creds or wrong region)
aws sts get-caller-identity
```

For boto3 inside scripts run locally, set the profile/region explicitly:

```python
import boto3
session = boto3.Session(profile_name="tfc", region_name="us-east-1")
```

> Note: this applies to CLI/boto3 run from this workstation. Code executed **inside
> SageMaker Studio notebooks** uses the notebook's execution role and the domain region —
> do not inject `--profile tfc` there.
