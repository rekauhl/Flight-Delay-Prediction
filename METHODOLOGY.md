# Methodology

This document explains every meaningful decision behind the project, what
was chosen, why, and what was tried and changed along the way.

## 1. The prediction boundary: Pre-Pushback vs. Post-Pushback

The single most important design decision in this project is **when** the
model is allowed to make its prediction.

- **Model 1 (Pre-Pushback):** every feature is something known before the
  aircraft leaves the gate (the schedule, the route, the distance, the carrier).
- **Model 2 (Post-Pushback):** identical feature set, plus `DepDelay` (how
  late the aircraft actually left the gate).

`DepDelay` is only known **after** pushback happens, so it is strictly excluded
from Model 1. It is deliberately included in Model 2. Both models share the exact same feature-building
code. The only difference between them is that one column. This means any
difference in their performance can be attributed cleanly to that one piece
of information, rather than to any other change in how the data was prepared.

Also excluded from **both** models, for the same reason as `DepDelay`, but with
no ambiguity about it: `DepTime`, `TaxiOut`, `TaxiIn`, `WheelsOff`, `WheelsOn`,
`ActualElapsedTime`, `AirTime`, and the delay-cause breakdown columns
(`CarrierDelay`, `WeatherDelay`, etc.). None of these are known before or even
right at pushback. They only exist once the flight is underway or has
landed (target leakage).

## 2. Data sourcing

**[Sourced]** US DOT Bureau of Transportation Statistics, "Reporting Carrier
On-Time Performance" table, downloaded directly from the public TranStats
PREZIP file server (one file per month).

Scope was narrowed to keep the project focused and the data volume manageable:
- **4 carriers:** American, Delta, United, Southwest
- **10 airports:** ATL, ORD, DFW, DEN, LAX, JFK, SFO, SEA, CLT, MSP
- **Hub-to-hub only:** both the origin **and** the destination must be in the
  airport list above. Flights connecting to a smaller airport outside this
  list are excluded entirely.

## 3. Why 2024 + 2025, and what didn't work first

The project originally used a single year (2025 only), split by month:
train on the first eight months, validate on the next two, test on the
last two.

This turned out to be broken in a specific way. A model trained only on
January–August has never seen a September, October, November, or December
flight. Because tree-based models cannot extrapolate to unseen feature ranges, 
this temporal split created an invalid extrapolation task rather than a reliable evaluation setup.

**The fix:** add a second year. Training on all of 2024 means every month is
genuinely represented before validation or test ever begins.

- **Train:** all of 2024
- **Validation:** January–June 2025
- **Test:** July–December 2025

This also fixed a side effect of the old design: the original single-year
split had noticeably different delay rates across train (23.9%), validation
(19.3%), and test (25.6%), because it accidentally split the year into a
naturally calmer stretch (Sep–Oct) versus busier ones. The new split is much
more balanced (22.2% / 22.4% / 24.3%).

**Limitation:** A year-by-year check (`MONTH` arrival delay rate in 2024 vs.
2025) showed the overall seasonal shape holds up well (a summer peak, a
fall dip, in both years), but a few months moved substantially in exact
magnitude, most notably **December: 19.1% in 2024 vs. 29.0% in 2025**. Since December sits in the test
period, the model has likely learned an outdated sense of how risky December
actually is. This shows up directly in the results (see Section 6) and is
treated as an accepted, named limitation rather than something to
over-engineer around.

## 4. Target definition

**[Sourced]** `ArrDel15` — BTS's own binary flag for "arrived 15+ minutes
late." Cross-checked against `ArrDelay >= 15` computed independently from the
raw minutes column: zero mismatches across all ~636,000 rows.

`ArrDelay` (the raw minutes-late column) is kept in the pipeline only for
this verification step and for exploratory analysis. It is never used as a
model input, and never used as an alternate target.

Cancelled and diverted flights are dropped before modeling. They have no
valid arrival delay to predict.

## 5. Features

| Feature | Type | Notes |
|---|---|---|
| `Reporting_Airline`, `Origin`, `Dest` | Categorical (one-hot encoded) | Airline, origin airport, and destination airport |
| `MONTH` | Categorical (one-hot encoded) | Month |
| `DAY_OF_WEEK` | Categorical (one-hot encoded) | Day of the week |
| `DEP_HOUR`, `ARR_HOUR` | Categorical (one-hot encoded) | Scheduled departure hour and arrival hour (departure hours 2–4 have zero flights in this dataset entirely) |
| `IS_HOLIDAY` | Binary | US federal holidays ± 1 day, via pandas' built-in holiday calendar |
| `CRSElapsedTime`, `Distance` | Numerical | Scheduled flight duration (minutes) and flight distance (miles) |
| `DepDelay` | Numerical | Actual departure delay in minutes (Model 2 only) |

Temporal features (`DEP_HOUR`, `ARR_HOUR`, `MONTH`, `DAY_OF_WEEK`) were one-hot encoded rather than 
kept as numericals because arrival delay rate is inherently non-linear. 
As shown in `README.md`, delays do not scale linearly from morning to night 
or month to month. Because Logistic Regression is a linear classifier, treating these features as raw integers 
forces it to fit a monotonic trend, effectively assuming hour 23 has 23× the linear 
impact of hour 1. One-hot encoding converts each time slot into an independent binary 
indicator, allowing linear models to learn distinct coefficients for each hour/day/month.

**[Assumption]** `IS_HOLIDAY` is a partial proxy, not a complete one. It
covers the holidays themselves and the day on either side, but it misses some
of the actual busiest travel days of the year. For example, the Sunday after
Thanksgiving isn't a federal holiday and falls outside the ±1 day window.

**Dropped from the feature set (mined already or not usable):**
`FlightDate`, `CRSDepTime`, `CRSArrTime` (all converted into the
features above before being dropped), `Cancelled`, `Diverted` (used only to
filter rows), `ArrDelay` (verification-only, see Section 4).

## 6. Modeling and evaluation

Four models were trained per configuration (Model 1 / Model 2), plus a baseline:

- **Naive baseline:** always predicts the training set's average arrival delay rate.
  Computed once since it doesn't use any features, so `DepDelay`'s presence or
  absence doesn't change it.
- **Logistic Regression:** standardized inputs (to treat all features fairly regardless of scale), tuned over
  `class_weight` (unweighted vs. balanced) using validation performance.
- **Random Forest:** tuned over tree depth, minimum leaf size, and class
  weight (12 combinations) using validation performance.

Model 1 and Model 2 are tuned **independently**: Model 2 does not reuse
Model 1's winning settings. This costs a bit of extra compute but avoids a
real risk: `DepDelay` is such a dominant feature that the **ideal** model
settings for Model 2 could plausibly be different, and reusing Model 1's
settings could have quietly undersold Model 2's true performance.

**Primary metric: PR-AUC.** With only ~23% of flights delayed, plain
accuracy is misleading. A model that always guesses "on time" already
scores 77%. PR-AUC focuses specifically on how well the model tells real
delays apart from false alarms. ROC-AUC is reported alongside it as a more
familiar reference point.

**Decision threshold:** chosen using the validation set's precision-recall
curve, targeting ~80% recall (catching 4 out of 5 real delays). This reflects
that missing a real delay is more costly than a false alarm, for a use case
like proactive rebooking or gate planning. This threshold is then applied to the
test set.

All four tuned models hit almost exactly 80% recall on the validation set
they were tuned on, by construction. On the test set, Model 2's recall holds
up well (~79–80%). Model 1's does not (~65%), which is consistent with the
year-to-year `MONTH` drift described in Section 3, since Model 1 relies more
heavily on month-based features than Model 2 does.

## 7. Explicitly out of scope (considered, not built)

- **Per-route/per-carrier historical delay rate:** considered in detail,
  including a tiered fallback for routes with too little data to trust on
  their own. Set aside in favor of first testing whether static features
  (carrier, airport, time, month) were sufficient on their own, which they
  largely were, based on feature importance results.
- **Aircraft rotation continuity** (how late an aircraft's previous flight
  that day was): a genuinely informative pre-pushback signal, but adds real
  data-quality complexity (tail number reliability, handling the first flight
  of an aircraft's day) for a project already covering its main ideas without
  it.
- **Weather data:** a plausible source of real signal, particularly for the
  month-level effects described in Section 3, but not pulled in for this
  version.
