# Hotel Booking Cancellation — Project Writeup

**Author:** Sai  
**Completed:** 2026-05  
**Dataset:** [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) — 119,390 bookings, 32 columns, 2015–2017  
**GitHub:** https://github.com/karthi9407/hotel-booking-cancellation

---

## 1. Problem Framing

After a booking is confirmed, the hotel wants to predict whether that reservation is likely to cancel. By estimating this early, hotels can manage their room inventory better and account for expected cancellations. This means we can only use information available at the moment of booking — any data that is only known after booking confirmation will likely introduce leakage into the model.

---

## 2. Data

The dataset contains 119,390 bookings across two hotel types — City Hotel (79,330) and Resort Hotel (40,060) — covering arrivals from 2015 to 2017. 37% of bookings were cancelled, making the dataset imbalanced and accuracy a misleading metric. Two columns had significant missing values: `company` at 94% and `agent` at 14% — rather than dropping them, we converted them to binary flags because the missingness itself is signal: a missing company means the booking wasn't corporate, a missing agent means the guest booked directly. The most counterintuitive finding was that non-refundable deposit bookings had a much higher cancellation rate than refundable ones — the opposite of what you'd expect, likely driven by OTAs booking in bulk speculatively and cancelling unused rooms.

---

## 3. Leakage Audit

We dropped three columns before modeling. `reservation_status` and `reservation_status_date` directly reveal the outcome — they are filled in after the cancellation happens, so using them as features would mean the model is reading the answer rather than predicting it. `assigned_room_type` is only known at check-in, not at the time of booking. By removing these, we ensure the model is only predicting from what the hotel actually knows at the moment a reservation is made.

---

## 4. Train / Val / Test Split

We used a time-based split rather than random shuffling because dates play a significant role — booking behaviour and cancellation rates change by season. Training on a future booking and validating on a past one would not reflect how the model is actually used. We trained on 2015–2016, validated on January–May 2017, and tested on June–August 2017. The test AUC dropped from 0.835 to 0.796 because the test set covers peak summer travel, which has higher booking volumes and different cancellation patterns than the training period — a distribution shift the model hadn't fully seen.

---

## 5. Feature Engineering

We created `is_corporate` and `booked_via_agent` as binary flags because `company` and `agent` were mostly missing — the absence of a value is itself the signal, indicating a direct or non-corporate booking. We bucketed `lead_time` into ranges rather than using the raw number to handle extreme outliers (some bookings were made 2 years in advance). We also created `total_guests` as a simple sum of adults, children, and babies. All preprocessing — scaling and one-hot encoding — was wrapped in a sklearn Pipeline, which ensures the scaler and encoder are fit only on training data and applied to val and test, preventing leakage from validation data into the preprocessing step.

---

## 6. Models & Results

| Model | Val ROC AUC | Notes |
|-------|------------|-------|
| Majority class baseline | 0.500 | Always predicts not cancelled |
| Logistic regression (numeric only) | 0.730 | No categorical features |
| Logistic regression (full features, balanced) | 0.820 | Categorical features added significant signal |
| Random Forest | 0.810 | Slightly below LR — linear relationships were strong enough that LR had an edge |
| XGBoost | 0.835 | Best val AUC; scale_pos_weight for class imbalance |
| **XGBoost — final test** | **0.796** | **4-point drop from val; test set is peak summer (distribution shift)** |

We tried three models: logistic regression, random forest, and XGBoost. XGBoost performed best with a validation AUC of 0.835. On the final test set (June–August 2017), AUC dropped to 0.796, likely because the test set covers summer peak season which has different cancellation patterns than the training period.

---

## 7. Threshold Tuning

The model outputs a probability between 0 and 1 for each booking — not a direct yes or no. The threshold is where you draw the line between predicting cancellation and not. The default of 0.5 is arbitrary, so we tuned it to 0.379 by maximising F1 on the validation set. At this threshold, the model catches 79% of actual cancellations but flags some bookings incorrectly — 42% of predicted cancellations turn out to be guests who show up. A hotel prioritising customer experience should raise the threshold to reduce false alarms; one prioritising revenue protection from empty rooms should lower it.

---

## 8. What I'd Do Next

I would invest more in feature engineering — specifically exploring better encoding for high-cardinality columns like `country` and `reserved_room_type`, which we one-hot encoded but may carry more signal with target encoding. I would also go deeper on evaluation metrics — understanding not just ROC AUC but also PR AUC and probability calibration, so the model's confidence scores are meaningful for business decisions like how aggressively to overbook. Finally, I would better define the business case upfront — for example, building separate models for peak summer and off-peak seasons, since the test set drop showed that a single model struggles with distribution shift across seasons.

---

## 9. Key Lessons

Understanding the data — its patterns, completeness, and quirks — is the most important step before any modeling begins. EDA isn't a formality; it's where you find the problems that will break your model later. Implementing a model felt intimidating before this project, but it turned out to be mostly code you can apply once you understand what each piece does. The harder and more valuable skill is interpreting the results and being able to explain them clearly to someone who wasn't in the room when you built it. And most importantly — understanding the business problem first shapes every decision that follows, from what features to use to which errors are acceptable.
