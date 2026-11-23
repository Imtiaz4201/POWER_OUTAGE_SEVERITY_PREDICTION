# Outage Severity Predictor (Flask)

A small web app for our final project. At the moment a major power outage starts, it predicts:

- **customers affected** (count, and per 100,000 residents),
- **outage duration** (minutes),
- **chance of a severe outage** (calibrated %), with a **Severe / Not severe** decision.

The user enters a **state**, the **outage start time** and the **top 5 features** (pre-filled with typical values for the state). The other model features (110 numeric + 5 categorical, 115 in total) are filled automatically. Time features come from the start time. The rest are the typical values (median, or most common category) of recent outages (2019–2023) in the same state with the same storm status.

## Models (from `final_stage2`)

| Target | Model (selected by CV in Experiment 2) | Test 2022–2023 |
|---|---|---|
| Customers affected (log per 100k) | Random forest | R² 0.49 (CV 0.31) |
| Duration (log minutes) | Random forest | R² 0.12 |
| Severe (top 25% customers or duration) | Logistic regression + isotonic calibration with prior correction (Experiment 7) | ROC-AUC 0.78 |

- **Setup:** strict storm rule, scenario A (pre-event information only), trained on outages before 2022.
- **Decision rule:** "severe" if the probability is above 1 / (1 + c), where c = cost of a missed severe outage relative to one false alarm.

**Top 5 inputs.** These are the five features with the highest random-forest importance (Experiment 5), normalised per target and summed over the three targets.

1. `hist_state_storm_cust`: typical customers affected in past outages in the state (same storm status)
2. `nws_watch_active`: NWS watches in effect
3. `noaa_n_zones`: zones with an active storm in the last 24 h
4. `nws_hours_since_warning`: hours since the last NWS warning
5. `noaa_storm_count`: storm reports active in the last 24 h

## Repository layout

```
.                                        repo root
├── app/                                 the deployed web app (see below)
│   ├── application.py                   Flask app (Elastic Beanstalk entry point: `application`)
│   ├── predictor.py                     loads the .pkl files, fills the features, predicts
│   ├── models/                          customers_model.pkl, duration_model.pkl,
│   │                                    severe_model.pkl, app_metadata.pkl
│   ├── templates/, static/              page, CSS, JS
│   ├── tests/                           test_app.py (smoke test)
│   ├── training/                        train_model.py (rebuilds the .pkl files; not deployed)
│   ├── requirements.txt                 runtime dependencies (no XGBoost / TensorFlow)
│   ├── Procfile                         gunicorn command for Elastic Beanstalk
│   └── Dockerfile                       for App Runner / ECS
│
├── final_stage2/                        final pipeline and experiments (source of the models)
│   ├── build/                           final data build
│   ├── exp1_walk_forward/               Experiment 1: walk-forward evaluation      (results/)
│   ├── exp2_model_search/               Experiment 2: model search and CV          (results/)
│   ├── exp3_learning_curve_ablation/    Experiment 3: learning curve, ablation     (results/)
│   ├── exp4_per_target_selection/       Experiment 4: per-target model selection   (results/)
│   ├── exp5_feature_selection/          Experiment 5: feature importance, top 5    (results/)
│   ├── exp6_error_diagnosis/            Experiment 6: error diagnosis              (results/)
│   └── exp7_rq2_reliability/            Experiment 7: severe-class calibration     (results/)
│
└── progress_stage2/                     earlier progress work and report
    ├── report_stage2/                   report (figures/)
    ├── results_outage_extended_noaa_nws_newgrid/
    └── results_outage_old_noaa_hifld/
```

Only `app/` is deployed. `app/__pycache__/` is generated automatically and can be ignored.

## Run locally

All commands below run **from inside the `app` folder**, so `cd app` first (from the repo root):

```bash
cd app
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python application.py            # open http://127.0.0.1:8000
python tests/test_app.py         # smoke test
```

**JSON API** (with the app running):

```bash
curl -X POST http://127.0.0.1:8000/api/predict -H "Content-Type: application/json" -d '{
  "state": "TEXAS", "start": "2023-06-15T14:30", "cost_ratio": 1,
  "hist_state_storm_cust": 600, "nws_watch_active": 2, "noaa_n_zones": 5,
  "nws_hours_since_warning": 3, "noaa_storm_count": 8 }'
```

## Deploy on AWS (EC2, Learner Lab)

Deploys the app on one Amazon EC2 server in AWS Academy Learner Lab and ends with the app open at `http://<public DNS>:8080/`. The server runs the `app/` folder from this repo directly.

**Why these settings:**

- **Instance type t3.small (2 GiB).** The random-forest models need several hundred MB of memory; t3.micro may run out of memory.
- **Python 3.11.** Amazon Linux 2023 ships Python 3.9, but `requirements.txt` (scikit-learn 1.7.2, numpy 2.1.3, scipy 1.15.2) needs Python 3.10+, and the `.pkl` models were saved with these exact versions. Always use `python3.11 -m pip`, never `pip3`.
- **Port 8080.** The app's own default port is 8000, so gunicorn is started on 8080 to match the security group rule.

### 1. Launch the EC2 server (AWS Console)

1. **Start the lab.** Open https://www.awsacademy.com/vforcesite/LMS_Login → **Student Login** → your **AWS Academy Learner Lab** course → **Modules** → **Launch AWS Academy Learner Lab** → **Start Lab**. Wait until the dot next to **AWS** is green, then click **AWS**.
2. **Open EC2.** Search for **EC2** and click **Launch instance**.
3. **Name and image.** Name: `outage-severity-app`. Choose **Amazon Linux** (Amazon Linux 2023 AMI, free tier eligible).
4. **Instance type.** Select **t3.small**.
5. **Key pair.** **Create new key pair**: name `outage-key`, type **RSA**, format **.pem**. Keep `outage-key.pem` **outside the repo** (for example in the folder above it) and never commit it.
6. **Security group.** In **Network settings** click **Edit** → **Create security group**. Keep rule 1 (SSH, port 22) and add rule 2: Custom TCP, port **8080**, source **Anywhere**.
7. **Launch.** Keep the default 8 GiB gp3 storage and click **Launch instance**. Open the instance (ID starts with `i-`) and wait until it is **Running** with 3/3 status checks.
8. **Copy the Public DNS** (like `ec2-3-84-197-37.compute-1.amazonaws.com`). You need it below.

### 2. Connect and install Python 3.11

On the instance page click **Connect** → **EC2 Instance Connect** → **Connect** (user `ec2-user`). In the black terminal:

```bash
sudo su                                          # become root, prompt changes to #
dnf install python3.11 python3.11-pip -y         # wait for "Complete!"
python3.11 --version                             # must print Python 3.11.x
```

Keep this tab open.

### 3. Upload the app from your laptop

Open a **new** Terminal (Mac) or PowerShell (Windows), not the EC2 tab. Go to the **repo root** (the folder that contains `app/`) and run the commands one at a time, using **your own Public DNS**. This example has `outage-key.pem` in the folder above the repo.

Mac:

```bash
cd /path/to/repo
chmod 400 ../outage-key.pem
scp -r -i ../outage-key.pem app ec2-user@<your-public-DNS>:~/
```

Windows (PowerShell):

```powershell
cd C:\path\to\repo
icacls ..\outage-key.pem /inheritance:r
icacls ..\outage-key.pem /grant:r "$($env:USERNAME):(R)"
scp -r -i ..\outage-key.pem .\app ec2-user@<your-public-DNS>:~/
```

If asked "Are you sure you want to continue connecting?", type `yes`. The upload (about 26 MB) takes under a minute.

### 4. Install the libraries (on the server)

Back in the EC2 Instance Connect tab (the `#` prompt):

```bash
cd /home/ec2-user/app
rm -rf training __pycache__                      # not needed to run; keep tests/ for the smoke test
ls                                               # Dockerfile Procfile README.md application.py models predictor.py requirements.txt static templates tests
python3.11 -m pip install -r requirements.txt    # 2-4 minutes; a pip-as-root warning is normal
```

### 5. Run and test

```bash
python3.11 tests/test_app.py                     # optional smoke test: expect "all tests passed"

nohup python3.11 -m gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 4 --timeout 60 application:application > app.log 2>&1 &
```

The second command prints a line like `[1] 3683`. **Write down that number**: it is the process ID you need to stop the app.

Wait about 10 seconds for the models to load, then check from the server:

```bash
curl http://localhost:8080/health                # must return {"status":"ok"}
```

Open `http://<your-public-DNS>:8080/` in your browser. Use **http://**, not https://. Choose a state (the five inputs fill in automatically), pick a start time and a cost ratio, then click **Predict**.

Optional JSON API test (laptop or server):

```bash
curl -X POST http://<your-public-DNS>:8080/api/predict -H "Content-Type: application/json" -d '{"state": "TEXAS", "start": "2023-06-15T14:30", "cost_ratio": 1, "hist_state_storm_cust": 600, "nws_watch_active": 2, "noaa_n_zones": 5, "nws_hours_since_warning": 3, "noaa_storm_count": 8}'
```

### 6. Stop everything so no credit is wasted

1. **Stop the app** (use the process ID from step 5): `kill 3683`. If you lost it, `pkill -f gunicorn` works too.
2. **Close the terminal:** type `exit` twice (root, then ec2-user).
3. **Terminate the instance:** EC2 console → **Instances** → tick `outage-severity-app` → **Instance state** → **Terminate (delete) instance** → confirm. Wait until the state is **Terminated**.
4. **End the lab:** in the first browser tab click **End Lab** (the dot next to AWS turns red).

To come back later, repeat from step 1. The Public DNS is new every time, and a new instance starts empty, so the upload and install steps must be repeated.

### Other options

The `app/` folder also includes a `Procfile` (Elastic Beanstalk, `eb init -p python-3.11 ...` then `eb create ... --instance_types t3.small --single`) and a `Dockerfile` (App Runner / ECS, port 8000, 1 vCPU / 2 GB). The health-check URL is `/health`.

## Retrain the models

The models need the raw data in `po1/Data` and the `tf_cpu` environment. Run from the **repo root**:

```bash
/opt/anaconda3/envs/tf_cpu/bin/python app/training/train_model.py
```

This reruns the final-stage data pipeline (`final_stage2`) and refits the three models with the Experiment 2 settings. It then rewrites `app/models/*.pkl` and prints the test scores as a check.

## Limits

The predictions are early estimates, not exact values. Duration is hard to predict (R² 0.12). The model also tends to over-predict small non-weather events (for example attacks), because the cause is not known when the outage starts.
