# Predictive Maintenance Studio — Idea Phase Submission

**Theme:** Agentic Predictive Maintenance Studio (Theme 1)
**Repo:** https://github.com/DaCameraGirl/abb-predictive-maintenance-2026

## The problem

Bearings are one of the most common failure points in the motors and
drives ABB builds, and they rarely fail without warning: vibration
characteristics shift measurably before a bearing physically breaks.
The gap isn't sensor data, most rotating equipment is already
instrumented, it's turning that raw vibration signal into a maintenance
decision a technician can trust and act on before the failure happens,
not after.

Two things make that harder than it sounds. First, most public
"predictive maintenance" demos train and validate on simulated data,
which sidesteps the parts of the problem that make it hard in the
field: you rarely have hundreds of independent identical assets that
all failed the same way, and you have to be honest about what you do
and don't know when an asset hasn't failed yet. Second, a maintenance
model that's wrong in an overconfident way, telling a technician "you
have three more weeks" when the real number is three more days, is
worse than useless. It's a system nobody should trust with real
equipment.

## What we built

A remaining-useful-life (RUL) prediction pipeline trained and validated
on the NASA/IMS bearing test-to-failure dataset: four real bearings on
one motor shaft, monitored by real accelerometers every 10 minutes for
about a week, until one of them (bearing 1) actually failed with a
documented outer race defect. This is real physical degradation data,
not a simulation.

The pipeline:

- **Feature extraction** from raw 20kHz vibration snapshots into
  standard vibration-analysis statistics (RMS, kurtosis, skew,
  peak-to-peak, crest factor) per bearing per snapshot.
- **An XGBoost regressor** predicting RUL in snapshots (and hours) from
  those features.
- **A dashboard** with a time slider over the full ~7-day test, a live
  risk list across all four bearings, a per-bearing SHAP breakdown, and
  a time-ordered chart of predicted vs. actual RUL across bearing 1's
  whole life.

## Leakage-safe validation

Our first validation attempt used a random split across bearing 1's
snapshot history, and it produced a great-looking RMSE. It was also
wrong: nearby-in-time snapshots are nearly identical, so a random split
leaks the shape of the near future across the train/test boundary. That
number was optimistic in a way that would have mattered the moment this
was judged against real held-out data.

We replaced it with a **chronological, expanding-window walk-forward
backtest**: the model only ever trains on data strictly earlier than
what it's asked to predict, rolled forward across several folds. This
is slower to compute and reports honestly worse numbers, but it's the
only validation methodology that actually tests forecasting rather than
interpolation. We report:

- MAE and RMSE, in both snapshots and hours
- An **asymmetric score** that penalizes late predictions (the model
  saying "more time left" than a bearing actually has) more heavily
  than early ones, because that's the direction that gets maintenance
  scheduled too late
- An **80% empirical uncertainty interval**, built from the spread of
  walk-forward backtest residuals. This is explicitly a **global range
  derived from historical residuals, not a per-prediction, conditionally
  calibrated interval** — it doesn't currently widen or narrow based on
  how unusual a given reading is. Getting there would mean quantile
  regression or a conformal-prediction layer; we scoped that out for
  now rather than ship something that looks more rigorous than it is.

The model actually serving live predictions in the dashboard is refit on
the complete trajectory after backtesting, the same way a real deployed
system would use all history collected so far. The backtest measures
honesty; the production model uses everything available.

## Explainability, and separating two different claims

Every prediction ships with a SHAP breakdown showing which vibration
features pushed the RUL estimate away from the model's average output,
in both snapshots and hours.

More importantly, the dashboard treats **"is this bearing degrading"**
and **"this bearing has approximately N snapshots left"** as two
separate claims with two separate reliability levels. The first is a
simple statistical deviation check against each bearing's own healthy
baseline, and it holds up even when the exact number is uncertain. The
second depends on a regression model generalizing from one observed
failure pattern. Collapsing them into a single confident-sounding number
would overstate what the system actually knows.

## Honest handling of censored data

Only bearing 1 failed during this test. Bearings 2, 3, and 4 were still
running when the recording stopped, they're **right-censored**: we know
they survived at least this long, but we don't know their true RUL. The
dashboard never presents their model output as a known value. Their
cards are explicitly labeled "no failure observed," their estimate is
flagged as extrapolated beyond the model's only observed failure label,
and marked not independently verifiable. This matters because it would
be easy, and dishonest, to quietly treat all four bearings the same way
in the UI.

## Business value

For ABB specifically: bearing failure in motors and drives is a
recurring, expensive, unplanned-downtime problem, and it's exactly the
class of failure this pipeline targets. The value isn't the specific
model trained on one 2004 test rig, it's the pattern: real vibration
data in, an honestly-validated remaining-life estimate out, an
explanation a maintenance engineer can sanity-check against their own
knowledge of the machine, and clear signaling about when the system is
confident versus extrapolating. That pattern transfers directly to
ABB's own instrumented equipment.

## Scope and limitations

This prototype is trained and backtested on **one confirmed failure
trajectory**. It demonstrates trajectory fitting and honest backtesting
methodology on real physical degradation data. It is **not** a
validated general bearing-failure model, and we're not claiming it
generalizes beyond the specific failure mode (outer race defect) and
rig it learned from. We're also not calling this an MVP: it's a scoped
technical prototype built to prove the methodology holds up under
scrutiny, not a minimum viable product aimed at shipping.

## Path to production

The single biggest limitation here is the single failure trajectory,
and the fix is more real data, not more modeling cleverness:

1. **Incorporate the other IMS test runs.** The same dataset includes
   two more test sets with additional real failures (inner race and
   rolling-element defects), giving multiple independent failure
   trajectories instead of one.
2. **Move to cross-run validation.** With multiple real failures, we can
   hold out entire failure runs the way the original design held out
   whole turbofan engines, real generalization testing across assets,
   not just across time within one asset.
3. **Calibrated, conditional prediction intervals**, once there's enough
   data to fit them properly (quantile regression or conformal
   prediction) instead of the current global residual-based range.
4. **Streaming ingestion** from live accelerometer feeds instead of
   static files, with the same feature extraction and degradation
   signal running continuously.
5. **Model monitoring and retraining triggers** as new failure events
   accumulate, so the model's confirmed-failure sample size actually
   grows over time in deployment, closing the gap this submission is
   upfront about.
