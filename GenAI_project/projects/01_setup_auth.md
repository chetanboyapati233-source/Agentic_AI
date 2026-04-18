# 01 — Setup & Auth

Goal: have a Python environment that can call Vertex AI Gemini from your machine.

---

## 1. Create / pick a GCP project

```bash
# Create (skip if you already have one)
gcloud projects create ai-eng-labs-$(date +%s)
# Set it as default
gcloud config set project YOUR_PROJECT_ID
# Link billing (needed for Vertex AI)
# → console.cloud.google.com/billing
```

## 2. Enable APIs

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  bigquery.googleapis.com \
  storage.googleapis.com \
  dlp.googleapis.com \
  run.googleapis.com
```

## 3. Create a bucket (for later weeks)

```bash
gsutil mb -l us-central1 gs://YOUR_PROJECT_ID-ai-labs
```

## 4. Authenticate locally

For dev work on your laptop:

```bash
gcloud auth login
gcloud auth application-default login
```

This creates Application Default Credentials (ADC). The Vertex AI SDK will pick them up
automatically — no API key, no service-account JSON file needed for local development.

## 5. Verify the Python client

```bash
pip install -r requirements.txt
python - <<'PY'
import vertexai
from vertexai.generative_models import GenerativeModel
import os

vertexai.init(project=os.environ["PROJECT_ID"], location=os.environ.get("LOCATION", "us-central1"))
model = GenerativeModel("gemini-1.5-flash-002")
resp = model.generate_content("Say 'hello data engineer' in five languages.")
print(resp.text)
PY
```

If this prints five greetings, you're done.

---

## Common setup gotchas

- **"Permission denied"** — your user account needs `roles/aiplatform.user` on the project. On a personal project the project owner already has it.
- **"Billing not enabled"** — Vertex AI requires an active billing account linked to the project.
- **Region mismatch** — if you created resources (e.g., buckets, indexes) in one region, use the same region for Vertex AI calls. `us-central1` is a safe default.
- **Quota** — the first time you call Gemini Pro you may hit a 0-quota error. Request a quota increase in the console; it's usually auto-approved.

## When to switch to a service account

Use a service account with a narrow role (not owner) when:
- You're deploying code to Cloud Run, Cloud Functions, Cloud Build, etc.
- You're sharing code with a teammate and don't want to rely on their user creds.
- You want to simulate production auth patterns in Week 11.

```bash
gcloud iam service-accounts create ai-eng-sa \
  --description="AI Engineer lab service account"
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:ai-eng-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

Running on a GCP compute resource? Attach the SA to it. Running locally? Use
`gcloud auth application-default login` with your user account for dev, and create a key
only if you must (it's a liability — rotate frequently).
