"""
12 — Supervised fine-tuning on Vertex AI (Week 10)

Goal:
  - Prepare a JSONL dataset in the Vertex AI SFT format.
  - Submit a supervised tuning job for Gemini Flash.
  - Poll for completion, then call the tuned endpoint.

Key intuition:
  Only fine-tune when prompting+RAG can't get you there: style, tone, strict format,
  domain vocabulary. Fine-tuning does NOT give the model new facts.

COST WARNING:
  Tuning jobs are single-digit to low-double-digit dollars per run. Read pricing first.

Run:
  python 12_finetuning_supervised.py prepare    # build dataset + upload to GCS
  python 12_finetuning_supervised.py tune       # submit tuning job
  python 12_finetuning_supervised.py use <endpoint>   # chat with the tuned model
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Iterator

from google.cloud import storage
from vertexai.preview.tuning import sft
from vertexai.generative_models import GenerativeModel

from _config import init_vertex, PROJECT_ID, LOCATION, STAGING_BUCKET, GEMINI_MODEL


# ---------- Dataset ----------
# Task: a strict "JSON data-ops incident summary" style. Prompting gets us ~70% — we'll
# push to near-100% with 50 examples.

EXAMPLES = [
    {
        "prompt": "Airflow DAG 'billing_etl' failed twice in the last hour with OOM.",
        "completion": (
            '{"service":"billing_etl","severity":4,"category":"oom",'
            '"action":"increase executor memory and retry"}'
        ),
    },
    {
        "prompt": "BigQuery export to customer_exports bucket has 0 rows for 3 days.",
        "completion": (
            '{"service":"bq_export_customers","severity":5,"category":"data_freshness",'
            '"action":"investigate upstream source and backfill"}'
        ),
    },
    {
        "prompt": "dbt model 'daily_revenue' has a null sum check failing.",
        "completion": (
            '{"service":"dbt_daily_revenue","severity":3,"category":"data_quality",'
            '"action":"inspect source tables for nulls in amount column"}'
        ),
    },
    # In a real run, you'd have 50–200 more examples like these. We keep the script
    # short for the lab; edit this list and re-run `prepare`.
]


def prepare():
    """Write Vertex AI SFT JSONL and upload to GCS."""
    assert STAGING_BUCKET.startswith("gs://"), "Set STAGING_BUCKET in .env"
    bucket_name = STAGING_BUCKET.removeprefix("gs://").split("/")[0]
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)

    local = Path("/tmp/sft_train.jsonl")
    with local.open("w") as f:
        for ex in EXAMPLES:
            row = {
                "contents": [
                    {"role": "user",  "parts": [{"text": ex["prompt"]}]},
                    {"role": "model", "parts": [{"text": ex["completion"]}]},
                ]
            }
            f.write(json.dumps(row) + "\n")
    blob = bucket.blob("ai_eng_labs/sft_train.jsonl")
    blob.upload_from_filename(local)
    uri = f"gs://{bucket_name}/ai_eng_labs/sft_train.jsonl"
    print(f"Uploaded {len(EXAMPLES)} examples to {uri}")
    return uri


def tune():
    training_uri = f"gs://{STAGING_BUCKET.removeprefix('gs://').split('/')[0]}/ai_eng_labs/sft_train.jsonl"
    print(f"Submitting tuning job with training_dataset={training_uri}")
    job = sft.train(
        source_model=GEMINI_MODEL,          # e.g. gemini-1.5-flash-002
        train_dataset=training_uri,
        tuned_model_display_name="incident-summarizer-v1",
        epochs=3,
        adapter_size=4,
        learning_rate_multiplier=1.0,
    )
    print(f"Job submitted: {job.resource_name}")
    print("Polling every 60s until complete...")
    while not job.has_ended:
        time.sleep(60)
        job.refresh()
        print(f"  state: {job.state}")
    print(f"Tuned model endpoint: {job.tuned_model_endpoint_name}")
    print("Use that with the `use` subcommand.")


def use(endpoint: str):
    model = GenerativeModel(endpoint)
    for prompt in [
        "Kafka consumer lag on topic 'orders' is 5M and growing.",
        "Terraform apply failed for prod-bq-dataset module with IAM permission denied.",
    ]:
        print("\nPROMPT:", prompt)
        print("TUNED:", model.generate_content(prompt).text)


if __name__ == "__main__":
    init_vertex()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    if cmd == "prepare":
        prepare()
    elif cmd == "tune":
        tune()
    elif cmd == "use":
        use(sys.argv[2])
    else:
        print(__doc__)


# ===== Try next =====
# 1. Run the same prompts through the BASE model and compare JSON cleanliness side-by-side.
# 2. Add 50 more examples and re-tune. Does quality plateau? When?
# 3. Use your Week 9 eval harness to score base vs tuned on a held-out set.
# 4. Tune Gemini Pro instead of Flash. Compare cost/quality; pick the winner.
