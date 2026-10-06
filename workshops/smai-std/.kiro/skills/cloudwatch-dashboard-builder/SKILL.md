---
name: cloudwatch-dashboard-builder
description: Ready-made CloudWatch dashboard JSON and CLI commands for SageMaker endpoint monitoring. Use when asked to create or automate the SageMaker-ML-Operations dashboard, SNS alert topic, or CloudWatch alarms (latency, error rate) via the AWS CLI instead of the console.
---

# CloudWatch Dashboard & Alarm Builder for SageMaker Endpoints

Automates Lab 5A's dashboard and alarms with the AWS CLI. Substitute `<ENDPOINT_NAME>`, `<REGION>`, and `<EMAIL>` before running. Discover the endpoint first:

```bash
aws sagemaker list-endpoints --query 'Endpoints[?EndpointStatus==`InService`].EndpointName' --output text
```

**Identity note:** `cloudwatch:PutDashboard` is not included in `AmazonSageMakerFullAccess`, so these commands may fail from a SageMaker Studio terminal (execution role). Run them from CloudShell or any shell using the Workshop Studio participant role. `put-metric-alarm` works from either.

## 1. SNS topic for alerts

```bash
TOPIC_ARN=$(aws sns create-topic --name SageMaker-Alerts --query TopicArn --output text)
aws sns subscribe --topic-arn "$TOPIC_ARN" --protocol email --notification-endpoint <EMAIL>
echo "Confirm the subscription from your email inbox before alarms can notify you."
```

## 2. Dashboard: SageMaker-ML-Operations

```bash
ENDPOINT=<ENDPOINT_NAME>
REGION=<REGION>
cat > /tmp/dashboard.json <<EOF
{
  "widgets": [
    {
      "type": "metric", "x": 0, "y": 0, "width": 12, "height": 6,
      "properties": {
        "title": "Endpoint Latency (microseconds)", "region": "$REGION",
        "stat": "Average", "period": 60, "view": "timeSeries",
        "metrics": [
          ["AWS/SageMaker", "ModelLatency", "EndpointName", "$ENDPOINT", "VariantName", "AllTraffic"],
          [".", "OverheadLatency", ".", ".", ".", "."]
        ]
      }
    },
    {
      "type": "metric", "x": 12, "y": 0, "width": 12, "height": 6,
      "properties": {
        "title": "Invocations (Sum, 5 min)", "region": "$REGION",
        "stat": "Sum", "period": 300, "view": "singleValue",
        "metrics": [
          ["AWS/SageMaker", "Invocations", "EndpointName", "$ENDPOINT", "VariantName", "AllTraffic"]
        ]
      }
    },
    {
      "type": "metric", "x": 0, "y": 6, "width": 12, "height": 6,
      "properties": {
        "title": "Error Rate (%)", "region": "$REGION", "period": 300, "view": "timeSeries",
        "metrics": [
          ["AWS/SageMaker", "Invocation4XXErrors", "EndpointName", "$ENDPOINT", "VariantName", "AllTraffic", {"id": "m1", "stat": "Sum", "visible": false}],
          [".", "Invocation5XXErrors", ".", ".", ".", ".", {"id": "m2", "stat": "Sum", "visible": false}],
          [".", "Invocations", ".", ".", ".", ".", {"id": "m3", "stat": "Sum", "visible": false}],
          [{"expression": "100*(m1+m2)/m3", "label": "Error Rate (%)", "id": "e1"}]
        ]
      }
    },
    {
      "type": "metric", "x": 12, "y": 6, "width": 12, "height": 6,
      "properties": {
        "title": "Endpoint Resource Utilization (%)", "region": "$REGION",
        "stat": "Average", "period": 60, "view": "timeSeries", "stacked": true,
        "metrics": [
          ["/aws/sagemaker/Endpoints", "CPUUtilization", "EndpointName", "$ENDPOINT", "VariantName", "AllTraffic"],
          [".", "MemoryUtilization", ".", ".", ".", "."]
        ]
      }
    }
  ]
}
EOF
aws cloudwatch put-dashboard --dashboard-name SageMaker-ML-Operations --dashboard-body file:///tmp/dashboard.json
```

Note: instance-level `CPUUtilization`/`MemoryUtilization` live in the **`/aws/sagemaker/Endpoints`** custom namespace (not `AWS/SageMaker`), and CPU is per-core scaled (400% = 4 vCPUs fully used).

## 3. Alarm: high latency

Threshold is in **microseconds**: 200 ms = 200000.

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name High-Endpoint-Latency-Alert \
  --alarm-description "Endpoint latency above 200ms" \
  --namespace AWS/SageMaker --metric-name ModelLatency \
  --dimensions Name=EndpointName,Value=$ENDPOINT Name=VariantName,Value=AllTraffic \
  --statistic Average --period 300 \
  --evaluation-periods 3 --datapoints-to-alarm 2 \
  --threshold 200000 --comparison-operator GreaterThanThreshold \
  --treat-missing-data notBreaching \
  --alarm-actions "$TOPIC_ARN"
```

## 4. Alarm: error rate > 5% (metric math)

```bash
cat > /tmp/error-rate-metrics.json <<EOF
[
  {"Id": "e1", "Expression": "100*(m1+m2)/m3", "Label": "Error Rate (%)", "ReturnData": true},
  {"Id": "m1", "ReturnData": false, "MetricStat": {"Stat": "Sum", "Period": 300, "Metric": {"Namespace": "AWS/SageMaker", "MetricName": "Invocation4XXErrors", "Dimensions": [{"Name": "EndpointName", "Value": "$ENDPOINT"}, {"Name": "VariantName", "Value": "AllTraffic"}]}}},
  {"Id": "m2", "ReturnData": false, "MetricStat": {"Stat": "Sum", "Period": 300, "Metric": {"Namespace": "AWS/SageMaker", "MetricName": "Invocation5XXErrors", "Dimensions": [{"Name": "EndpointName", "Value": "$ENDPOINT"}, {"Name": "VariantName", "Value": "AllTraffic"}]}}},
  {"Id": "m3", "ReturnData": false, "MetricStat": {"Stat": "Sum", "Period": 300, "Metric": {"Namespace": "AWS/SageMaker", "MetricName": "Invocations", "Dimensions": [{"Name": "EndpointName", "Value": "$ENDPOINT"}, {"Name": "VariantName", "Value": "AllTraffic"}]}}}
]
EOF
aws cloudwatch put-metric-alarm \
  --alarm-name High-Error-Rate-Alert \
  --alarm-description "Endpoint error rate above 5%" \
  --metrics file:///tmp/error-rate-metrics.json \
  --evaluation-periods 2 --datapoints-to-alarm 2 \
  --threshold 5 --comparison-operator GreaterThanThreshold \
  --treat-missing-data notBreaching \
  --alarm-actions "$TOPIC_ARN"
```

## 5. Verify

```bash
aws cloudwatch get-dashboard --dashboard-name SageMaker-ML-Operations --query DashboardName
aws cloudwatch describe-alarms --alarm-names High-Endpoint-Latency-Alert High-Error-Rate-Alert \
  --query 'MetricAlarms[].{Name:AlarmName,State:StateValue}'
```

`INSUFFICIENT_DATA` right after creation is normal; with `notBreaching` missing-data handling, alarms settle to `OK` once evaluated.

## Cleanup (end of Lab 5 only — 5B/5C reuse the SNS topic)

```bash
aws cloudwatch delete-alarms --alarm-names High-Endpoint-Latency-Alert High-Error-Rate-Alert
aws cloudwatch delete-dashboards --dashboard-names SageMaker-ML-Operations
aws sns delete-topic --topic-arn "$TOPIC_ARN"
```
