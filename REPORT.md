# Assignment 1 Report

- **Name:** Ainhoa Olaechea
- **Student ID:** 18680
- **Email:** aolaechea.ieu2022@student.ie.edu
- **Group:** A

## Dataset

I used the **Hotel Booking Demand** dataset, downloaded from Kaggle (`hotel_bookings.csv`).
It comes from a 2019 research paper by Antonio, Almeida & Nunes and contains real
bookings from two hotels in Portugal: a City Hotel (Lisbon) and a Resort Hotel (Algarve).

- **One row = one room booking**, with what the hotel knew about it (lead time, length of
  stay, number of guests, meal plan, price per night, booking channel, guest country,
  previous cancellations, etc.) and whether it was eventually cancelled.
- **Size:** 119,390 rows × 32 columns, arrivals from July 2015 to August 2017.
- **Target:** `is_canceled` (1 = cancelled, 0 = the guest arrived). 37% of all bookings
  were cancelled: 41.7% at the City Hotel and 27.8% at the Resort Hotel.

**Why I picked it:** cancellations are a real and expensive problem for hotels, the
target is clearly defined, and the dataset has several traps (columns only known after
the booking, time ordering, duplicate group bookings) that make feature engineering and
the split decision important rather than trivial.

**What I used:** only the City Hotel (79,330 bookings). I chose it because its demand is
much less seasonal than the Resort Hotel's: its median price only moves between about
€80 and €120 per night across the year, while the Resort goes from €48 in January to €188
in August. With weaker seasonality, the cost of a cancellation is roughly the same in any
month, so I can use one fixed cost per type of error instead of a different cost for
every season. I cleaned the data and randomly sampled it down to 20,000 bookings to stay
within the size guideline, using a fixed random seed (42) so that the same 20,000 bookings
are selected every time the pipeline runs. The sample's cancellation rate (41.8%) matches
the full City Hotel data (41.7%), so it is representative.

## Business / real-life framing

**Scenario:** the City Hotel loses revenue when guests cancel late, because the room often
stays empty. The model supports a **deposit policy**:

| | |
|---|---|
| **Goal** | Reduce revenue lost to cancelled bookings and keep occupancy high |
| **Model** | Predicts, *at the moment a booking is made*, the probability it will be cancelled |
| **Action** | Bookings above a risk threshold must pay a deposit, refundable only up to 14 days before arrival |

**Why the City Hotel:** a real model is built for one client, and the two hotels behave
very differently. The City Hotel has the bigger cancellation problem and steadier prices
through the year (about €80–€120/night vs €48–€188 at the Resort), so one fixed cost per
type of error is a reasonable approximation.

**Why a 14-day refund deadline:** in the City Hotel data, 12% of cancellations happen within
7 days of arrival, 19% within 14 days and 33% within 30 days. Two weeks gives the hotel time
to resell the room while being less strict than 30 days. One consequence: 18.5% of guests
book less than 14 days ahead, so for them the deposit is non-refundable from the start.

**What this framing implies for the pipeline:**

1. **Target definition.** I kept `is_canceled` (any cancellation). This is simple, but
   under my policy an early cancellation costs the hotel nothing (the deposit is refunded),
   while the model treats it the same as a late one. Predicting only late cancellations
   would match the policy more closely (see Limitations).
2. **Time-based split:** The hotel would train on past bookings and use the model
   on future ones. A random split would put 2017 bookings in training and test on 2015–16,
   letting the model "see the future" and overstating its performance. I split by
   arrival date: oldest 70% train, next 15% validation, newest 15% test.
3. **Leakage rule.** Every input must be known on the day of booking.
4. **`deposit_type` removed.** The model's job is to *decide* who pays a deposit, so it
   should not be given the deposit decision the hotel already made.
5. **Decision threshold set by business cost** The two errors cost different
   amounts. With a typical booking worth about €265 (median €100/night × 3 nights):
   - **Missed cancellation (false negative) ≈ €50:** only ~19% of cancellations fall inside
     the 14-day window where money is lost, so 0.19 × €265 ≈ €50.
   - **False alarm (false positive) ≈ €25:** I assume ~10% of reliable guests asked for a
     deposit would book elsewhere, so 0.10 × €265 ≈ €25.

   Because a missed cancellation costs twice as  much as a false alarm, the theoretical best threshold is below 0.5:
   25 / (25 + 50) ≈ 0.33. I choose the threshold on the validation set by minimising
   the total cost, and the dashboard lets you move it and see the cost change.

## Data preparation & feature engineering

### Data quality

- **Missing values:** `children` (4 rows) → 0; `country` (488 rows) → "Unknown" (kept as its
  own group rather than deleting the bookings); missing `agent`/`company` means the booking
  was made without one, so it becomes a yes/no feature.
- **Removed 432 broken rows:** 0 nights, 0 guests, or a price above €1,000/night (one booking
  was €5,400/night, clearly an error). Impact: 0.5% of City Hotel rows.
- **Kept 25,902 duplicate rows.** Without a booking ID, identical rows most likely represent
  several rooms in the same group booking. Each is a real room that can sit empty, and they
  cancel 61% of the time vs 29% for the rest, so removing them would hide an important part
  of the problem. Because I split by date and copies share the same arrival date, they end
  up on the same side of the split and cannot inflate the test score.

### Leakage: columns removed because they are not known at booking time

| Column | Evidence / reason |
|---|---|
| `reservation_status` | Matches the target perfectly ("Canceled"/"No-Show" ⇔ `is_canceled = 1`): it *is* the answer |
| `reservation_status_date` | Date of cancellation or check-out, which is in the future |
| `assigned_room_type` | Room given at check-in |
| `booking_changes` | Counts changes made after booking (always 0 on booking day) |
| `required_car_parking_spaces` | **0% cancellations among 492 bookings**: too perfect; most likely recorded at arrival |
| `total_of_special_requests` | 55% cancel with no requests vs ~22% with one or more; requests can be added after booking, so I removed it to be safe |

`days_in_waiting_list` was **kept**: it is known when the booking is confirmed, which is
exactly when the deposit decision is made.

`deposit_type` was removed for framing reasons (above). Interestingly, "Non Refund"
bookings cancel 99.8% of the time; most are groups and tour operators, who likely reserve
blocks of rooms and release the ones they don't use.

### Engineered features

Every feature is computed from a single row, so nothing from the test period can leak in.

| Feature | Evidence (cancellation rate) | Why |
|---|---|---|
| `total_nights` | 1 night 30%; 2–3 nights 47%; 8+ nights 54% | Length of stay as one number |
| `has_agent`, `has_company` | Company bookings 21% vs 43% | IDs are labels, not quantities (agent 240 is not "more" than agent 9) |
| `cancelled_before` | 94% vs 38% | Past behaviour predicts future behaviour |
| `log_lead_time` | ≤7 days 12% … >180 days 65% | Lead time is very skewed (0 to 600+ days); the log makes early differences matter more |
| `arrival_weekday` | Thursday 49% vs Sunday 34% | Business vs leisure travel patterns |
| `country`: top 10 + "infrequent" | Portugal 66% vs abroad 27% | 133 countries is too many for 20k rows; the Portugal column acts as a "domestic guest" flag |
| `revenue_at_risk` = price × nights | Weak alone | Links the model to the money at stake |

I did not add a separate `is_domestic` column because it would be an exact copy of the
Portugal column, and duplicated columns confuse a linear model.

**Removed date columns:** `arrival_date_year` (the test period is later, so the model would
meet a year it never saw), `arrival_date_week_number` (redundant, and treated as a quantity
it makes week 52 "bigger" than week 1), `arrival_date_day_of_month` (no business meaning;
weekday captures what matters) and `arrival_date_month` (weak effect, 36–46%, consistent with
the City Hotel not being very seasonal).

### Split and preprocessing

| Split | Rows | Arrival dates | Cancellation rate |
|---|---|---|---|
| Train | 14,000 | 2015-07-01 → 2017-02-18 | 41.3% |
| Validation | 3,015 | 2017-02-19 → 2017-05-24 | 45.1% |
| Test | 2,985 | 2017-05-25 → 2017-08-31 | 41.0% |

- The cut is made on dates, not row numbers, so all bookings arriving on the same day stay
  together.
- `StandardScaler` for numeric columns (mean 0, std 1, essential for gradient descent) and
  `OneHotEncoder` for categories (fewer than 20 occurrences → "infrequent"; country limited
  to the top 10). Both are fitted on the training set only, then applied to validation
  and test. Result: 20 features → 59 input columns.
- Validation is used to choose C, the learning rate and the threshold. The test set is used
  once, at the end.

## Modeling: three implementations, one model

**Model: logistic regression.** The target is binary, so logistic rather than linear
regression. It outputs a probability (not just yes/no), which the hotel can turn into a
deposit decision with a cost-based threshold, and its weights are interpretable.

**The three implementations, trained on the same 59 columns and the same split:**

1. **scikit-learn** `LogisticRegression`: C tried in {0.01, 0.1, 1, 10}; validation log loss
   0.4918 / 0.4759 / **0.4745** / 0.4746 → **C = 1**. Results from 0.1 to 10 are almost
   identical, so the model is not overfitting much.
2. **Manual PyTorch loop:** weights and bias created by hand (all 0), forward pass
   (`X @ w + b`, then sigmoid), binary cross-entropy written by hand plus the same L2
   penalty as sklearn (rescaled to `1 / (C·n)` because I use the mean loss instead of the
   sum), `loss.backward()`, and a manual gradient-descent update. Full batch, 2,000 epochs.
   Learning rate tried in {0.1, 0.5, 1.0}; validation log loss 0.4766 / 0.4747 / **0.4745**
   → **lr = 1.0** (0.1 was too slow to converge in 2,000 epochs).
3. **Standard PyTorch:** `nn.Module` with one `nn.Linear` layer, `BCEWithLogitsLoss` and
   `torch.optim.SGD`, with the same C, learning rate and number of epochs. Weights start at
   0 so the comparison with the manual version is fair. The L2 penalty is added to the loss
   by hand instead of using `weight_decay`, because `weight_decay` would also penalise the
   bias, which sklearn does not.

Both PyTorch versions start at a training loss of 0.6931 (= ln 2, a 50/50 guess, because
all weights start at 0) and end at 0.4332. All models and the fitted preprocessor are saved
to `models/`; the evaluation and the dashboard only load them.

### Results on the test set (2,985 bookings, 25 May – 31 Aug 2017)

| Model | Accuracy @0.5 | ROC-AUC | Log loss | Business cost @0.5 |
|---|---|---|---|---|
| Naive baseline (always predicts the training rate, 41.3%) | 0.590 | 0.500 | 0.677 | €61,200 (never asks) |
| scikit-learn | 0.728 | 0.809 | 0.523 | **€29,400** |
| Manual PyTorch | 0.727 | 0.809 | 0.523 | €29,525 |
| Standard PyTorch | 0.727 | 0.809 | 0.523 | €29,525 |

For context: asking every guest for a deposit costs €44,025, which is the best the
baseline can do with a cost-based threshold. The model cuts the cost by about 52% compared
with never asking, and saves about €14,600 compared with asking everyone.

### Do the three agree?

- **Manual vs standard PyTorch: yes, practically identical.** The largest weight difference
  is 0.000005 and they make the same decision for 100% of bookings. They run the same
  algorithm from the same starting point; the tiny difference comes from the `eps = 1e-7`
  in my hand-written loss versus PyTorch's numerically stable formula.
- **PyTorch vs scikit-learn: same predictions, different weights for rare categories.** On
  validation, probabilities differ by 0.0017 on average and 99.8% of decisions are the
  same. But some weights differ a lot, for example `distribution_channel_infrequent`:
  +1.39 in sklearn vs +0.26 in PyTorch.

  **Investigation:** all the large differences are in categories with very few training
  rows (1, 2, 12 and 35 bookings). With so few examples, the loss is almost flat in those
  directions, so plain gradient descent moves very slowly there and has not finished after
  2,000 epochs. sklearn's L-BFGS optimiser uses curvature information and converges further.
  So the difference comes from the **optimiser, not the model**, and it barely affects
  predictions because almost no booking has those values.
- **Consequence for interpretation:** the second-largest sklearn weight is based on only
  **2 bookings**, so it is noise, not a business insight. The reliable signals are
  Portuguese guests (+1.77), a previous cancellation (+1.34) and booking far in advance
  (+0.86); direct bookings cancel less (−0.96).

### Dashboard

`uv run python main.py app` opens a Gradio dashboard that only loads the saved models:
(1) the three-model comparison with a predicted-vs-actual (calibration) plot and probability
histograms; (2) target and feature distributions; (3) a threshold slider that updates the
confusion matrix and the business cost in euros, with a cost-vs-threshold curve compared to
"never ask" and "always ask".

## Limitations & next steps

| Limitation | What I found | How I would address it |
|---|---|---|
| **Drift over time** | On the test period the model predicted 46% of bookings would cancel, but only 41% did; ROC-AUC fell from 0.849 (validation) to 0.809 (test). The calibration plot sits below the diagonal | Retrain on a rolling window (e.g. the last 12 months) every season, and monitor predicted vs actual cancellation rate each month |
| **The validation threshold did not carry over** | The cheapest threshold on validation was 0.29, but on test it cost €29,950, slightly *more* than 0.5 (€29,400); the test curve is lowest around 0.4. I did not re-tune on test, as that would make the final score dishonest | Choose the threshold with rolling (walk-forward) validation over several periods instead of one, so it is less sensitive to one season |
| **Test set is summer only** | The test score reflects high season, not the whole year | Use walk-forward evaluation across several test windows covering all seasons |
| **Target doesn't match the policy exactly** | Early cancellations cost nothing under my policy but count the same as late ones | Build a new target: cancelled within 14 days of arrival or no-show (constructed from the cancellation date), and compare business cost |
| **Cost assumptions** | The 10% of guests who would leave if asked for a deposit is a guess | Run a small A/B test (ask a random group for deposits) to measure the real walk-away rate, then update the costs |
| **Possibly undetected leakage** | Removing parking and special requests was a judgement call based on suspicious patterns | Confirm with the hotel when each field is recorded, and keep only fields filled in at booking time |
| **Unreliable rare-category weights** | Weights based on 1–35 bookings | Raise `min_frequency`, or merge rare channels and segments into broader groups |
| **Group bookings counted per room** | 25,902 duplicates give large groups more weight | With a booking ID, model group bookings separately or weight them per booking |
| **Old data, one hotel** | 2015–2017 data from one Lisbon hotel; booking behaviour may have changed since | Retrain on recent data before using it, and test on another hotel before generalising |

## AI use disclosure

I used Claude through the assignment:
- to explain concepts (leakage, time-based splits, gradient descent, regularisation,
  calibration, business-cost thresholds);
- to write the code for data loading and cleaning, feature engineering, the three training
  implementations, saving and loading models, evaluation, and the Gradio dashboard;

The choices of client hotel, business action, 14-day deadline, features to build and
columns to drop were decided by me.