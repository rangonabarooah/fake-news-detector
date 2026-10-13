# Fake News Detection AI

Flask app that classifies news text as REAL or FAKE using TF-IDF and a Passive Aggressive classifier.

## Run locally

Requires Python 3.10 or newer.

```bash
cd fake-news-detector
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py                    # builds model.joblib from data/
python app.py                      # open http://127.0.0.1:5000
```

Stop the server with Ctrl+C.

## Data

`data/` should contain the Kaggle "Fake and Real News Dataset" files, `Fake.csv` and `True.csv`. `train.py` uses them automatically. Without them, it trains on a tiny built-in sample and the model will be inaccurate.

## Deploy on Render

1. Push this folder to a GitHub repo (keep `data/` and `model.joblib` included so the build can train, or let the build run `train.py`).
2. In Render: New → Blueprint, select the repo. `render.yaml` sets the build and start commands.
3. Build command: `pip install -r requirements.txt && python train.py`
4. Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`

## API

`POST /predict` with JSON `{"text": "..."}` returns `{"prediction": "REAL" | "FAKE", "score": number}`.

## Limitations

The model learns word patterns in the training data. It cannot check facts, and it may do poorly on news that differs in style or time period from the Kaggle data.
