# Lab 2 – Data Prep

**Persona:** Data Engineer

Prepare data for downstream model building using several approaches so you can pick the right tool for the workload. This lab covers two datasets:

- **Bank Marketing dataset** (UCI) — tabular dataset (~41K records, 20 features), used by Lab 3A (traditional ML with XGBoost)
- **Databricks Dolly 15K** — instruction-following dataset filtered for summarization, used by Lab 3B (LLM fine-tuning)

All options produce outputs in **Amazon S3**. For the Bank Marketing dataset, a shared Iceberg registration step persists the data into **S3 Table Buckets** and **Athena Iceberg tables** for downstream analytics, drift monitoring (Lab 5), and QuickSight dashboards.

## Data preparation options

| Lab | Dataset | Option | Best For |
|-----|---------|--------|----------|
| **2A** | Dolly 15K (Summarization) | In-line processing | Quick text dataset prep — filter, validate, split, and upload directly from notebook |
| **2B** | Bank Marketing | SageMaker Processing Job (scikit-learn) | Fully managed container — no cluster admin |
| **2C** | Bank Marketing | EMR Serverless | Serverless Spark — auto-scales, no capacity planning |
| **2D** | Bank Marketing | EMR on EC2 | Managed clusters — large/long-running jobs |
| **2E** | Bank Marketing | Canvas (Low-code/No-code) | Visual prep — GUI-based, no coding required |

## Workflow

### Lab 2A — Dolly dataset (LLM fine-tuning prep)

1. **Load** the Databricks Dolly 15K dataset
2. **Filter** for the summarization category
3. **Validate** — remove empty or too-short records
4. **Split** into training (70%) and evaluation (30%) sets
5. **Create prompt template** (`template.json`) for instruction tuning
6. **Upload to S3** — `train.jsonl`, `test.jsonl`, and `template.json`

### Lab 2B–2E — Bank Marketing dataset (traditional ML prep)

1. **Run one of Lab 2B–2E** — produces `train.csv` and `test.csv` in S3
2. **Run the Iceberg registration notebook** — registers the CSVs into:
   - S3 Table Bucket (managed Iceberg, auto-compaction)
   - Default Glue Catalog Iceberg tables (Athena, QuickSight, Lab 5)
3. **Capture snapshot IDs** — used by Lab 3 and Lab 5 for baseline pinning

```
Lab 2B/2C/2D/2E  →  train.csv + test.csv in S3
        ↓
lab-2-traditional-ml-iceberg-registration.ipynb
        ↓
S3 Table Bucket + Athena Iceberg tables (with snapshot IDs)
        ↓
Lab 3 (model build) / Lab 5 (drift monitoring) / QuickSight
```

## Outputs

### Lab 2A outputs (consumed by Lab 3B — LLM fine-tuning)

- `s3://<DataBucketName>/<ProfileName>/dolly_dataset/train.jsonl` — Training examples
- `s3://<DataBucketName>/<ProfileName>/dolly_dataset/test.jsonl` — Evaluation examples
- `s3://<DataBucketName>/<ProfileName>/dolly_dataset/template.json` — Prompt template for instruction tuning

### Lab 2B–2E outputs (consumed by Lab 3A — traditional ML)

**CSV (regular S3):**
- `s3://<default-bucket>/bank-marketing-lab/data/train/train.csv` — Training data (CSV, no header, target-first)
- `s3://<default-bucket>/bank-marketing-lab/data/test/test.csv` — Test data (CSV, no header, target-first)

**S3 Table Bucket (managed Iceberg):**
- Bucket: `bank-marketing-monitoring-<account-id>`
- Namespace: `bank_marketing`
- Tables: `training_data`, `evaluation_data`

**Default Catalog Iceberg (Athena / QuickSight / Lab 5):**
- Database: `bank_marketing`
- Tables: `training_data`, `evaluation_data`
- Queryable via: `SELECT * FROM bank_marketing.training_data LIMIT 10;`
- Supports time travel: `FOR VERSION AS OF <snapshot_id>`

## Notebooks

| File | Description |
|------|-------------|
| `lab-2a-fine-tuning-data-prep-processing-job.ipynb` | Dolly summarization dataset preparation |
| `lab-2b-traditional-ml-data-prep-processing-job.ipynb` | Bank Marketing dataset preparation (SageMaker Processing Job) |
| `lab-2-traditional-ml-iceberg-registration.ipynb` | **Common step**: Register processed CSVs into S3 Table Bucket + Athena Iceberg (run after any of 2B–2E) |

## Prerequisites

- Completed Lab 1 (environment setup)
- Access to the workshop S3 bucket (DataBucketName from CloudFormation outputs)
- SageMaker Studio JupyterLab space running

### Iceberg registration prerequisites (one-time, admin)

The `lab-2-traditional-ml-iceberg-registration.ipynb` notebook requires the following account/role setup. In AWS Workshop Studio these are typically pre-provisioned; Section 0 of the notebook automates them if needed:

1. **IAM permissions** on the SageMaker execution role: `s3tables:*`, `glue:*` (catalog/database/table), `athena:*`, `lakeformation:*`
2. **Lake Formation data lake administrator** — the SageMaker role must be an LF admin to create tables and grant permissions
3. **S3 Tables → Glue Data Catalog integration** — enabled once per account (S3 console → Table buckets → Enable integration). Without it, Athena/Lake Formation can't see the S3 Table Bucket namespace
