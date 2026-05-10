# Project 1 Plan — Hotel Booking Cancellation Prediction

**Dataset:** [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) (Kaggle)
**Target:** `is_canceled` (binary)
**Why this dataset:** travel domain, real messiness, classification, ~37% positive class (instructive imbalance), datetime features, leakage traps to learn.

## 2-week schedule (10–12 hrs total)

Each session is sized for ~60–90 min. Don't rush. The point is to do each piece *properly*, not fast.

### Week 1 — Setup, EDA, baseline

**Session 1 (90 min) — Environment + data load**
- [ ] Install `uv` (or `poetry`)
- [ ] Create project env, install: pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter
- [ ] Download the dataset to `data/raw/` (NEVER commit raw data; check `.gitignore`)
- [ ] Load with pandas, run `.info()`, `.describe()`, `.head()`
- [ ] Write 5 sentences in `notebooks/01-eda.ipynb` describing what's in the dataset

**Session 2 (90 min) — EDA part 1: univariate**
- [ ] Distribution of every column. Histograms for numerics, bar charts for categoricals.
- [ ] Missing-data audit: which columns, how much, which records
- [ ] Class balance check on `is_canceled`
- [ ] Identify *suspicious* values (e.g. `adr` (average daily rate) negatives, `babies > 5`)

**Session 3 (90 min) — EDA part 2: bivariate + target**
- [ ] For each feature, compare cancellation rate by feature value
- [ ] Datetime features: cancellation rate by month, lead time, day of week
- [ ] Note features that look most predictive
- [ ] Identify potential leakage features (think: which columns wouldn't be available *before* cancellation?)

**Session 4 (90 min) — Train/val/test split + baseline**
- [ ] Define a clean problem framing: "Given booking info available at booking time, predict `is_canceled`."
- [ ] Drop or document any leakage features
- [ ] Train/val/test split with rationale (random vs time-based — discuss in writeup)
- [ ] Baseline 1: predict the majority class. Baseline 2: simple logistic regression with no FE.
- [ ] Metrics: precision, recall, F1, ROC AUC, PR AUC. **Not accuracy** — it'll be misleading.

### Week 2 — Feature engineering, models, evaluation, write-up

**Session 5 (90 min) — Feature engineering**
- [ ] Encode categoricals (one-hot or target encoding — discuss tradeoffs in writeup)
- [ ] Datetime features: extract month, day-of-week, lead-time-buckets
- [ ] Drop or impute missing values (decide and document)
- [ ] Use sklearn `Pipeline` so preprocessing is fit on train, applied to val/test (no leakage!)

**Session 6 (90 min) — Model 1: logistic regression done right**
- [ ] Logistic regression with regularization
- [ ] `GridSearchCV` over a small grid (C values)
- [ ] Read off coefficients, interpret top features
- [ ] Compare against baseline

**Session 7 (90 min) — Model 2: random forest**
- [ ] Random forest classifier
- [ ] Feature importance plot
- [ ] Compare to logistic regression

**Session 8 (90 min) — Model 3: gradient boosting (XGBoost or LightGBM)**
- [ ] Install xgboost or lightgbm; train on processed data
- [ ] Light hyperparam tuning
- [ ] Compare all three models on val set

**Session 9 (60 min) — Final eval + threshold tuning**
- [ ] Pick the best model on val set; do *one* eval on test set (no peeking before)
- [ ] Plot ROC curve, PR curve, confusion matrix
- [ ] Discuss precision/recall tradeoff and pick a threshold appropriate to a business framing

**Session 10 (90 min) — Write-up + repo polish**
- [ ] Write `WRITEUP.md`: problem framing, data, splits, features, models, results, what you'd do next
- [ ] Clean README
- [ ] Make repo public on GitHub
- [ ] Tweet/LinkedIn-post the result (yes really — visibility compounds)

## Acceptance criteria
- [ ] Public GitHub repo
- [ ] Reproducible env (uv lock or requirements.txt)
- [ ] EDA notebook
- [ ] Feature engineering as a sklearn Pipeline (no leakage)
- [ ] At least 3 models compared
- [ ] Threshold-tuned final eval
- [ ] WRITEUP.md
- [ ] One public mention (LinkedIn or X)

## Things you'll learn (and that'll show up in interviews)
- Train/val/test discipline
- Why accuracy is a trap on imbalanced data
- ROC vs PR AUC, when each misleads
- sklearn `Pipeline` and why it matters
- Tree-based vs linear models — strengths and tradeoffs
- Threshold tuning as a deployment-time decision
- Leakage spotting — the single most useful skill in applied ML
