# Lab 3 – Model Build

**Persona:** Data Scientist

Use the **Amazon SageMaker AI (SMAI)** platform tools — training, JumpStart, and the Model Registry — to experiment with and build both traditional ML and LLM models, with governance (experiment tracking, lineage, and model registry) built in.

## Contents
- `lab3a_traditional_ml_experimenation.ipynb` – Traditional ML with **SMAI**. Trains a binary classifier on the UCI Bank Marketing dataset to predict term-deposit subscriptions using managed SMAI training.
- `lab-3b-fine-tuning.ipynb` – Fine-tune **Llama 3.2 3B** for text summarization with **SMAI** JumpStart, tracking parameters, metrics, and base-to-fine-tuned lineage for auditability and reproducibility.
- `lab-3c-model-evaluation-registration.ipynb` – Evaluate the fine-tuned model against the base model, create model cards with governance metadata, and register versions in the **SageMaker Model Registry** with approval workflows.
- `metrics.py` – Shared evaluation metric helpers.

## Outcome
Trained and evaluated models registered in the Model Registry, ready for automated deployment in Lab 4.
