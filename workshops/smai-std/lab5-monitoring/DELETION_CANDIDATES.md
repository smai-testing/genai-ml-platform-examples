# Lab 5 — Deletion Candidates

This workshop is a **simplified** version of
[aws-samples/sample-mlops-bestpractices](https://github.com/aws-samples/sample-mlops-bestpractices).
The only deltas from that reference are that this workshop **drops**:

- automated/scheduled drift runs,
- EventBridge schedules,
- CloudWatch alarms, and
- the SNS/SQS/Lambda drift-alert delivery plane.

Lab 5 monitoring runs **entirely inside the notebooks** — Evidently data/model
drift → written directly to Athena → logged to MLflow → read by QuickSight.
**No infrastructure is created from a notebook**: the Athena tables
(`inference_responses`, `monitoring_responses`, `ground_truth_updates`) are
provisioned by the workshop stack (`scripts/deploy-workshop.sh` →
`templates/4-inference-capture.yaml`).

The files below **have been deleted** (they were out of scope). This file is
kept as the record of what was removed and why. `load_baseline_from_registry`
was extracted from `lambda_drift_monitor.py` into the kept module
`src/drift_monitoring/baseline.py` (imported by lab5d) before deletion.

Recover any file with `git show <branch>@{1}:<path>` or `git checkout <prev-sha> -- <path>`.

## CloudFormation stacks (all replaced by the workshop stack)

| File | Reason |
|---|---|
| `cloudformation/drift-monitoring-infra.yaml` | Creates SNS topic, SQS results queue, writer Lambda, **5 CloudWatch alarms**, CloudWatch dashboard — all out of scope. |
| `cloudformation/sagemaker-mlflow-setup.yaml` | Companion stack that re-creates Athena tables in a `fraud_detection` DB + inference-logger Lambda + QuickSight/LakeFormation IAM. Superseded by the workshop stack (`bank_marketing`). |
| `cloudformation/deploy-drift-monitoring.sh` | Deploys the drift-monitoring stack above. |

## Drift-monitoring Lambda / scheduled-execution machinery (`src/drift_monitoring/`)

| File | Reason |
|---|---|
| `manage_drift_lambda.py` | Builds/deploys/schedules the drift-monitor Lambda (EventBridge). |
| `deploy_monitoring_writer.py` | Deploys the SQS→Athena monitoring-results writer Lambda. |
| `create_cloudwatch_monitoring.py` | Creates CloudWatch alarms/dashboard. |
| `create_monitoring_table.py` | Creates `monitoring_responses` from a script — replaced by CFN (`4-inference-capture.yaml`). |
| `Dockerfile.lambda` | Container image for the drift-monitor Lambda. |

## Extracted before deletion

| File | Reason |
|---|---|
| `lambda_drift_monitor.py` | **DELETED.** `load_baseline_from_registry()` (+ its two helpers) was moved into the kept module `src/drift_monitoring/baseline.py`, which lab5d imports. The Lambda handler, SQS-send, and CloudWatch-alarm code were out of scope and are gone. |

## Setup / scheduled-execution helpers (`src/setup/`)

| File | Reason |
|---|---|
| `setup_scheduled_inference.py` | Schedules recurring inference (EventBridge). |
| `setup_scheduled_batch_transform.py` | Schedules recurring batch transform (EventBridge). |
| `codebuild_image.py` | Builds the drift-monitor Lambda container image via CodeBuild. |
| `create_athena_tables.py` | Creates Athena tables from a script — replaced by CFN (`4-inference-capture.yaml`). |

## Kept (the in-notebook monitoring path — do NOT delete)

`monitor_model_performance.py`, `evidently_reports.py`,
`generate_drift_dataset.py`, `simulate_ground_truth_from_athena.py`,
`update_ground_truth.py`, `log_monitoring_to_mlflow.py`,
`src/governance/create_governance_dashboard.py`, `src/utils/*`,
`src/train_pipeline/athena/athena_client.py`, `src/config/*`.
