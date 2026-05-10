# Hotel Booking Cancellation Prediction

Predicts whether a hotel booking will be cancelled using information available at the time of booking — enabling hotels to make smarter overbooking and inventory decisions.

**Final model:** XGBoost — ROC AUC 0.796 on held-out test set (Jun–Aug 2017)  
**Dataset:** [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) — 119,390 bookings, 2015–2017

---

## What I built

An end-to-end ML pipeline: EDA → feature engineering → three models compared → threshold tuning → final evaluation.

Key decisions made along the way:
- **Time-based train/val/test split** — bookings are temporal, so random splitting would leak future data into training
- **Leakage audit** — dropped columns only available after the booking outcome (reservation status, assigned room type)
- **Class imbalance handling** — 37% cancellation rate; used balanced class weights and ROC AUC instead of accuracy
- **Threshold tuning** — chose 0.379 instead of default 0.5 to catch more cancellations, with explicit business tradeoff documented

## Most interesting finding

Non-refundable bookings had a much higher cancellation rate than refundable ones — the opposite of intuition. Likely driven by OTAs booking rooms speculatively and cancelling unused inventory.

## Models compared

| Model | Val ROC AUC |
|-------|------------|
| Majority class baseline | 0.500 |
| Logistic regression (numeric only) | 0.730 |
| Logistic regression (full features) | 0.820 |
| Random Forest | 0.810 |
| XGBoost | **0.835** |
| XGBoost — final test | **0.796** |

## How to run

```bash
conda create -n ml-pipeline python=3.11 -y
conda activate ml-pipeline
pip install -r requirements.txt
jupyter lab
```

Download `hotel_bookings.csv` from Kaggle and place in `data/raw/`. Then open `notebooks/01-eda.ipynb`.

## Full writeup

See [WRITEUP.md](WRITEUP.md) for detailed explanation of every decision — data, leakage audit, feature engineering, model choices, threshold tuning, and key lessons.
