---
inclusion: always
---

# Lab 5 Monitoring — Shared Workshop Context

You are assisting a participant of the "GenAI/ML Standardization with SageMaker AI" workshop, working on Lab 5 (Monitoring). Persona: ML/Ops Engineer. Goal: monitor deployed models across operational health (5A), logs (5B), and API audit trails (5C).

## Environment facts

- **Account**: AWS Workshop Studio event account, provisioned by the workshop
  CloudFormation stack (`scripts/deploy-workshop.sh` → `templates/main.yaml`).
  ProjectName defaults to `bank-marketing-prediction`.
- **Region**: confirm the active region before running CLI commands or opening
  console links — it must match the region you deployed the workshop stack in.
  A resource ARN in another region means something is misconfigured.
- **Console identity**: the Workshop Studio participant role has `AdministratorAccess`. Console-based work (CloudWatch, CloudTrail, SNS, EventBridge) is not permission-constrained.
- **Notebook identity**: code in SageMaker Studio runs as the user-profile execution role (`{ProjectName}-usera-role` or `-userb-role`), which has `AmazonSageMakerFullAccess` plus scoped policies. Note: its `events:*` permission is limited to `us-east-1` rules; CloudWatch Logs access is read-only. Prefer the console for creating alarms, dashboards, metric filters, and EventBridge rules.
- **Domain**: SageMaker domain `{ProjectName}-domain`, VpcOnly mode. User profiles `userA` and `userB`. JupyterLab space `usera-jl-space`.

## Artifacts produced by earlier labs (what to monitor)

| Artifact | Naming pattern | Produced by |
|---|---|---|
| XGBoost training jobs | `bank-marketing-xgboost-*` | Lab 3A |
| Real-time endpoint | `bank-marketing-<YYYYMMDDHHMMSS>` | Lab 3A |
| MLflow experiment | `bank-marketing-prediction` | Lab 3A |
| Registered model | `bank-prediction-XGBoostModel` | Lab 3A |
| Llama fine-tuning jobs / endpoints | `jumpstart-*` | Lab 3B / 4B |
| Model package groups + approval events | project-specific | Lab 3C / 4A |

If the participant has no live endpoint (deleted for cost reasons), historical metrics for completed training jobs are still viewable; endpoints can be redeployed from the Lab 3A notebook (`lab3-model-build/lab3a_traditional_ml_experimenation.ipynb`).

## Guardrails

- **Never delete SageMaker endpoints, models, model package groups, or the MLflow app** unless the participant explicitly asks — later labs depend on them.
- Alarms, dashboards, SNS topics, metric filters, and EventBridge rules created in Lab 5 are disposable; help clean them up when asked.
- Do not create new SageMaker compute (endpoints, training jobs) to "generate data" — reuse existing artifacts or send inference traffic to existing endpoints instead.
- Keep costs minimal: no data-event CloudTrail trails, no OpenSearch domains, no new instances.
- SNS email subscriptions require the participant to confirm via email before notifications arrive — remind them.

## Naming conventions for Lab 5 resources

- SNS topic: `SageMaker-Alerts`
- Dashboard: `SageMaker-ML-Operations`
- Alarm names: descriptive kebab/Pascal, e.g. `High-Endpoint-Latency-Alert`
- Custom metric namespace: `SageMaker/CustomMetrics`
