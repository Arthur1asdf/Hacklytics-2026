# Survival Drivers Dashboard (No Gemini)

## What this adds

- Frame-level CSV logging from `cvopen.py`
- A Streamlit dashboard that visualizes:
  - Critical pressure timeline
  - Limb state timeline
  - State transition heatmap
  - Top risk windows

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Generate data

Run detector normally:

```bash
python3 cvopen.py
```

This writes one CSV per run to:

- `cv/logs/limb_run_YYYYMMDD_HHMMSS.csv`

## 3) Launch dashboard

```bash
streamlit run dashboard_survival.py
```

## Notes

- Data logging toggle is in `cvopen.py`:
  - `ENABLE_DATA_LOGGING = True`
- Logs are append-free: each run creates a new file.
