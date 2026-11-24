# Exploring Data Analysis

A Google Colab notebook walking through the Data Visualization & Analysis (DVA) lab —
**Experiments 1 to 7**. Everything lives in a single notebook, [`DVA.ipynb`](DVA.ipynb),
built so that *every step is one cell*: run it top to bottom and each output lands
directly beneath the step that produced it.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sania0507/Exploring-Data-Analysis/blob/main/DVA.ipynb)

## Contents

| Exp | Topic | Dataset |
| --- | --- | --- |
| 1 | Analysing and visualising datasets with pandas, NumPy, seaborn & matplotlib | `CARS.csv` (Kaggle `ljanjughazyan/cars1`) |
| 2 | Exploratory data analysis of the California Housing dataset | `california_housing_train.csv` (built into Colab) |
| 3 | Hotel booking demand — data quality, cancellations, monthly reservations | `datasetforexp3_hotel_booking.csv` (Kaggle hotel booking demand) |
| 4 | Time series analysis of the Daily Delhi Climate dataset | `DailyDelhiClimateTrain.csv` (Kaggle `sumanthvrao/daily-climate-time-series-data`) |
| 5 | Time series decomposition techniques | synthetic series (generated in-notebook) |
| 6 | Linear regression | `datasetforexp6_salary.csv` (`age`, `salary`) |
| 7 | k-Nearest Neighbours classification | inline dictionary (no CSV) |

### Experiment by experiment

**1 — Data analysis & visualisation fundamentals.** Starts from a brand-new Colab
notebook and a CSV upload, then works through the full inspection workflow:
`df.shape` and column names for instances vs features, dropping irrelevant columns
(`Drive Train`, `Model`, `Invoice`, `Type`, `Origin` — none help predict price),
`info()` for dtypes, `drop_duplicates` on `MSRP`, `isnull().sum()` for missing values,
and the five-number summary via `describe()`. Ends with a bar chart of cars per make
and a scatter of `Horsepower` vs `MSRP`, the two most related variables.

**2 — California Housing EDA.** Loads the file Colab ships under `/content/sample_data/`,
then profiles it: structure and dtypes, summary statistics, a KDE histogram of the
target `median_house_value`, and a geographic scatter plotting longitude against
latitude coloured by house value and sized by population — which reproduces the shape
of California. Closes with a correlation heatmap and a sorted list of correlations
against the target.

**3 — Hotel booking EDA.** Answers a set of set lab questions. Adds a unique `id`
column first, then defines `check_data(df)` to report dtype, null count, null
percentage, and unique count per column in one table. From there: cancellation counts
and proportions, cancellation rate split by `City Hotel` vs `Resort Hotel`, a
`df_checkout` frame of non-cancelled bookings, and reservations per arrival month
with month names mapped to numbers so they sort chronologically.

**4 — Climate time series.** The real indexing lesson: `to_datetime`, `set_index`,
`sort_index`, then `asfreq('D')` to rebuild a complete daily calendar so gaps surface
as `NaN`. Plots mean temperature, humidity, wind speed, and pressure, then computes
7-day and 14-day rolling means and overlays them on the raw series. Finishes with a
`PeriodIndex` check and a monthly `resample('ME').mean()` summary.

**5 — Decomposition.** Generates a 3-year daily series from a known formula — linear
trend + yearly sine seasonality + Gaussian noise — so the decomposition has a ground
truth to recover. Runs `seasonal_decompose(model='additive', period=365)`, plots trend
and seasonal components separately, inspects residual statistics, and compares a
30-day simple moving average against an EMA with `alpha=0.1`.

**6 — Linear regression.** Fits `LinearRegression` on a synthetic `Y = 3.5X + noise`
dataset, plots the best-fit line, and prints slope and intercept to confirm the model
recovers the generating parameters. Then the exercise proper: the salary dataset, a
scatter to verify the relationship is actually linear, a fitted model on `age` →
`salary`, and predictions for an entered age plus a table of predictions for ages
25–60.

**7 — KNN classification.** Demonstrates KNN's sensitivity to `k` by classifying the
same new point with `k=1` and `k=5` and plotting both results side by side. Then the
exercise: a 15-customer `Age`/`Income`/`Bought` dataset, a labelled scatter, predictions
for three new customers, a held-out accuracy check, and a full decision-boundary
surface for `k=3` computed over a meshgrid.

## Running

The notebook is self-contained — no local setup required:

1. Click the **Open in Colab** badge above.
2. Run cells top to bottom (`Runtime → Run all` works too).
3. When a step needs a CSV, Colab prompts for an upload. Filenames must match the
   `PATH` set in the preceding configuration cell — **change `PATH` if Colab saves your
   file under a different name** (e.g. `Cars.csv` vs `CARS.csv`).

To run locally instead:

```bash
pip install jupyter pandas numpy seaborn matplotlib scikit-learn statsmodels
jupyter notebook DVA.ipynb
```

## Datasets

Only Experiment 2 needs no upload — `california_housing_train.csv` ships with every
Colab session. Experiments 1, 3, 4, and 6 require a manual upload; Experiment 5
generates its own data; Experiment 7 types its data in as a dictionary.

Kaggle sources are listed per-experiment in the table above. Download the CSV, then
upload it into the Colab session storage before running that experiment's load cell.

## Notes

- Experiment 6's prediction step calls `input()`, so it blocks waiting for a typed age.
  In Colab the prompt appears inline under the cell; in a local Jupyter kernel it is
  usually fine, but `Run all` in some frontends can stall here.
- Cells assume a fresh session. Running an individual later cell without its earlier
  imports will raise `NameError` — start from the top or re-run the import cells.
- Statistics use fixed seeds (`random_state=42`, `np.random.seed(42)`) wherever there is
  randomness, so results are reproducible.
