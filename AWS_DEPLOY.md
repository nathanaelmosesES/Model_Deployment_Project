# AWS Deployment Guide — SageMaker + S3 + EC2 (GUI Only)

**Flow:**
```
GitHub repo ──► SageMaker Notebook ──► train ──► S3 bucket
                                                      │
                                               EC2 pulls artifacts
                                                      │
                                               uvicorn serve_api
```

---

## Prerequisites

- AWS account with billing enabled
- GitHub repo cloned locally (already done)
- Your `data_C.csv` file ready to upload

---

## Part 1 — Create S3 Bucket

1. Go to **AWS Console → S3 → Create bucket**
2. **Bucket name:** `credit-scoring-artifacts` (must be globally unique, add your name/initials if taken)
3. **Region:** pick one and **remember it** — use the same region for everything in this guide
4. **Block Public Access:** leave all 4 checkboxes **ON** (default)
5. Everything else: leave default → **Create bucket**

Inside the bucket, create two folders manually:
- Click **Create folder** → name it `data` → Create
- Click **Create folder** → name it `artifacts` → Create

Upload your training data:
- Open the `data/` folder → **Upload** → add `data_C.csv` → **Upload**

---

## Part 2 — Create IAM Role for SageMaker

1. Go to **AWS Console → IAM → Roles → Create role**
2. **Trusted entity type:** AWS service
3. **Use case:** SageMaker → **SageMaker** (the first option) → Next
4. The policy `AmazonSageMakerFullAccess` is pre-attached — leave it
5. Click **Next**
6. **Role name:** `SageMakerExecutionRole` → **Create role**

Add S3 access to this role:
1. Find the role you just created → click it
2. **Add permissions → Attach policies**
3. Search `AmazonS3FullAccess` → check it → **Add permissions**

---

## Part 3 — Create IAM Role for EC2

1. Go to **IAM → Roles → Create role**
2. **Trusted entity type:** AWS service
3. **Use case:** EC2 → Next
4. **Add permissions → Attach policies:**
   - Search and check `AmazonS3ReadOnlyAccess`
5. **Role name:** `EC2S3ReadRole` → **Create role**

---

## Part 4 — Run Training on SageMaker

### 4.1 Create a Notebook Instance

1. Go to **AWS Console → SageMaker → Notebook → Notebook instances → Create notebook instance**
2. **Notebook instance name:** `credit-scoring-training`
3. **Notebook instance type:** `ml.t3.medium` (cheap, enough for training prep)
4. **IAM role:** Select `SageMakerExecutionRole` (created in Part 2)
5. Everything else: leave default → **Create notebook instance**
6. Wait until **Status** shows **InService** (~2–3 minutes)

### 4.2 Open JupyterLab

1. Click **Open JupyterLab** next to your notebook instance
2. In JupyterLab, click the **Terminal** icon (bottom of the launcher)

### 4.3 Clone Repo & Install Dependencies

In the terminal:

```bash
cd SageMaker
git clone https://github.com/nathanaelmosesES/Model_Deployment_Project.git
cd Model_Deployment_Project
pip install -r requirements-all.txt
```

### 4.4 Download Training Data from S3

```bash
aws s3 cp s3://credit-scoring-artifacts/data/data_C.csv data_C.csv
```

### 4.5 Run Training

```bash
python scripts/train_local.py --data data_C.csv --output-dir artifacts
```

Training will take several minutes. When done, `artifacts/model.joblib` and `artifacts/feature_schema.json` will exist locally in the notebook.

### 4.6 Upload Artifacts to S3

```bash
aws s3 cp artifacts/model.joblib s3://credit-scoring-artifacts/artifacts/model.joblib
aws s3 cp artifacts/feature_schema.json s3://credit-scoring-artifacts/artifacts/feature_schema.json
```

Verify in the AWS Console → S3 → your bucket → `artifacts/` folder — both files should appear.

### 4.7 Stop the Notebook (to avoid charges)

Go back to **SageMaker → Notebook instances** → select your instance → **Actions → Stop**.

---

## Part 5 — Launch EC2 and Serve the API

### 5.1 Launch EC2 Instance

1. Go to **AWS Console → EC2 → Instances → Launch instances**
2. **Name:** `credit-scoring-api`
3. **AMI:** Ubuntu Server 24.04 LTS (free tier eligible)
4. **Instance type:** `t3.small` (or `t3.micro` for testing)
5. **Key pair:** Create new → name it `credit-scoring-key` → **RSA / .pem** → Download and save the `.pem` file somewhere safe
6. **Network settings:**
   - Click **Edit**
   - Add a rule: **Type** = Custom TCP, **Port** = `8000`, **Source** = `0.0.0.0/0`
   - (HTTP port 80 is added by default — you can keep it)
7. **Advanced details → IAM instance profile:** Select `EC2S3ReadRole`
8. **Launch instance**

Wait until **Instance state** = `running` and **Status check** = `2/2 checks passed` (~1–2 minutes).

### 5.2 Connect to EC2

1. Select your instance → **Connect → EC2 Instance Connect → Connect**
   - This opens a browser-based terminal — no SSH client needed

### 5.3 Set Up the Server

In the browser terminal:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python & pip
sudo apt install -y python3-pip python3-venv git

# Clone repo
git clone https://github.com/nathanaelmosesES/Model_Deployment_Project.git
cd Model_Deployment_Project

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements-all.txt
```

### 5.4 Download Artifacts from S3

```bash
aws s3 cp s3://credit-scoring-artifacts/artifacts/model.joblib artifacts/model.joblib
aws s3 cp s3://credit-scoring-artifacts/artifacts/feature_schema.json artifacts/feature_schema.json
```

### 5.5 Start the API

```bash
uvicorn scripts.serve_api:app --host 0.0.0.0 --port 8000
```

### 5.6 Test the API

Get your EC2 **Public IPv4 address** from the EC2 console, then open in your browser:

```
http://<your-ec2-public-ip>:8000/docs
```

The Swagger UI should appear with the `/predict` endpoint ready to use.

---

## Part 6 — Keep API Running After Disconnect (Optional)

If you close the browser terminal, the API stops. To keep it running:

```bash
# Install screen
sudo apt install -y screen

# Start a persistent session
screen -S api

# Run the API inside screen
source .venv/bin/activate
uvicorn scripts.serve_api:app --host 0.0.0.0 --port 8000

# Detach from screen (API keeps running): press Ctrl+A then D
# Reattach later with: screen -r api
```

---

## Re-deploy After Retraining

Whenever you retrain (SageMaker → run training again → upload new artifacts to S3):

```bash
# On EC2 (reconnect via EC2 Instance Connect)
cd Model_Deployment_Project
git pull
aws s3 cp s3://credit-scoring-artifacts/artifacts/model.joblib artifacts/model.joblib
aws s3 cp s3://credit-scoring-artifacts/artifacts/feature_schema.json artifacts/feature_schema.json

# Restart API
screen -r api
# Ctrl+C to stop, then rerun uvicorn
```

---

## Cost Estimate

| Resource | Type | Estimated Cost |
|---|---|---|
| SageMaker Notebook | ml.t3.medium | ~$0.05/hr (stop when done) |
| S3 Storage | ~200 MB | <$0.01/month |
| EC2 | t3.small | ~$0.023/hr |

**Always stop SageMaker notebook instances when not training** — they bill by the hour even when idle.
