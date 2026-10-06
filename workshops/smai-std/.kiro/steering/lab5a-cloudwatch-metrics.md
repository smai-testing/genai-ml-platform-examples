---
inclusion: always
---

# Lab 5A — CloudWatch Operational Monitoring

Assist the participant in monitoring SageMaker training jobs and endpoints with CloudWatch metrics: explore metrics, build a dashboard, create alarms, and reason about cost optimization. Console-first; use CLI/boto3 only where it is faster and the identity allows it (see shared context).

## Lab flow (keep the participant on this path)

1. **Training job metrics** — namespace `/aws/sagemaker/TrainingJobs`, dimension `Host` (streams named like `<training-job-name>/algo-1`). Metrics: `CPUUtilization`, `MemoryUtilization`, `GPUUtilization`, `GPUMemoryUtilization`, `DiskUtilization`. Values can exceed 100% (per-core scale: e.g. 400% = 4 vCPUs fully used).
2. **Endpoint metrics** — two namespaces, participants confuse them:
   - `AWS/SageMaker` (dimensions `EndpointName, VariantName`): `Invocations`, `InvocationsPerInstance`, `ModelLatency`, `OverheadLatency`, `ModelSetupTime`, `Invocation4XXErrors`, `Invocation5XXErrors`. **ModelLatency/OverheadLatency are in MICROSECONDS** — 200 ms = 200,000. Flag this whenever thresholds are involved.
   - `/aws/sagemaker/Endpoints` (custom namespace): instance-level `CPUUtilization`, `MemoryUtilization`, `DiskUtilization`.
3. **Generate traffic** — endpoint metrics only exist after invocations. Reuse the Lab 3A notebook's `endpoint.invoke(body=test_csv, content_type='text/csv')` in a loop (10–20 requests, ~2 s apart), or `aws sagemaker-runtime invoke-endpoint` with a valid CSV row. Metrics appear after 2–3 minutes (up to 5).
4. **Dashboard** `SageMaker-ML-Operations` — widgets: ModelLatency (line), Invocations sum (number, 5-min period), error rate via metric math `(m1+m2)/m3*100`, endpoint CPU/Memory (stacked area).
5. **Alarms** to SNS topic `SageMaker-Alerts`:
   - `High-Endpoint-Latency-Alert`: ModelLatency avg > 200000 (µs), 5-min period, 2/3 datapoints.
   - `High-Error-Rate-Alert`: metric math error-rate > 5%, treat missing data as `notBreaching` (low-traffic lab endpoints have empty periods).
6. **Cost analysis** — compare utilization vs instance size; suggest right-sizing math. Do not actually resize or redeploy anything.

## Verification commands (read-only, safe to run)

```bash
# find the participant's endpoint and training jobs
aws sagemaker list-endpoints --query 'Endpoints[].{Name:EndpointName,Status:EndpointStatus}'
aws sagemaker list-training-jobs --sort-by CreationTime --sort-order Descending --max-results 10 \
  --query 'TrainingJobSummaries[].TrainingJobName'

# confirm invocation metrics are flowing (adjust names/times)
aws cloudwatch get-metric-statistics --namespace AWS/SageMaker --metric-name Invocations \
  --dimensions Name=EndpointName,Value=<endpoint> Name=VariantName,Value=AllTraffic \
  --start-time <ISO> --end-time <ISO> --period 300 --statistics Sum
```

## Common pitfalls to preempt

- "No metrics visible": endpoint never invoked, wrong region, or wrong namespace (`AWS/SageMaker` vs `/aws/sagemaker/Endpoints`). Check in that order.
- Latency alarm thresholds entered in ms instead of µs (or vice versa) — always restate units.
- SNS notifications not arriving: subscription unconfirmed (check email, including spam).
- Anomaly-detection alarms are not useful in this lab — the endpoint has minutes of history, not the days the model needs. Steer participants to static thresholds.
- Training-job metrics live under **Custom namespaces** in the console metric browser, not AWS namespaces — participants often search the wrong list.
- If the endpoint was deleted, don't recreate infrastructure ad hoc; point to the Lab 3A notebook deployment cells or analyze historical training-job metrics instead.

## Out of scope for 5A

Logs and Logs Insights (Lab 5B), CloudTrail/EventBridge auditing (Lab 5C), model quality/drift/bias monitoring (Labs 5D–5F). If asked, give a one-line answer and defer to the right lab.
