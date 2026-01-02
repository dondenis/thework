# %%
# JUPYTER CELL — 0) Setup & Load  (ADJUSTED)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

# ---- CONFIG ----
DATA_PATH = "sepsis_final_data_RAW_withTimes.csv"          # your main row-per-4h dataset
SEPSIS_PATH = "new_sepsis_mimiciii.csv"  # contains morta_in_hosp, max_sofa, max_sirs, sepsis_time

# Map your column names here if they differ
# COLUMN_MAP = {
#     "id": "m:icustayid",
#     "sex": "o:gender",                  # accepts 'F'/'M' or 0/1
#     "age_days": "o:age",                # will convert to years (÷365.25)
#     "intime": "m:intime",
#     "iv_4h": "o:input_4hourly",
#     "vp_max": "o:max_dose_vaso",
#     "block_idx": "o:bloc",
#     "charttime": "m:charttime"
# }

df = pd.read_csv(DATA_PATH)
# for alias, col in COLUMN_MAP.items():
#     if col in df.columns:
#         df.rename(columns={col: alias}, inplace=True)

# load 90-day mortality file
sepsis_df = pd.read_csv(SEPSIS_PATH)


# %% [markdown]
# ### Basic cleaning and derived fields

# %%
# JUPYTER CELL — 1) Basic Cleaning & Derived Fields  (ADJUSTED)
# Age -> years
if "o:age" in df.columns:
    df["age_years"] = pd.to_numeric(df["o:age"], errors="coerce") / 365.25
else:
    df["age_years"] = np.nan
df.loc[df["age_years"] <= 0, "age_years"] = np.nan

# Gender -> female flag
def to_female(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().upper()
    if s in {"F","FEMALE","1"}: return 1
    if s in {"M","MALE","0"}:   return 0
    try:
        v = float(s); return 1 if v > 0 else 0
    except:
        return np.nan
if "o:gender" in df.columns:
    df["female"] = df["o:gender"].map(to_female)
else:
    df["female"] = np.nan

# JUPYTER CELL — 1) Basic Cleaning & Derived Fields  (PATCHED MORTALITY BLOCK)

# Merge morta_in_hosp info (and related fields) from new_sepsis_mimiciii.csv
if "sepsis_time" in sepsis_df.columns:
    sepsis_df["sepsis_time"] = pd.to_datetime(sepsis_df["sepsis_time"], errors="coerce")

merge_cols = [c for c in ["icustayid","morta_inhosp","max_sofa","max_sirs","sepsis_time"] if c in sepsis_df.columns]
df = df.merge(sepsis_df[merge_cols], left_on="m:icustayid", right_on="icustayid", how="left")

# %% [markdown]
# #### ICU Hours

# %%
# Ensure numeric (UNIX seconds)
df["m:charttime"] = pd.to_numeric(df["m:charttime"], errors="coerce")
df["m:presumed_onset"]    = pd.to_numeric(df["m:presumed_onset"], errors="coerce")

# Per patient: max charttime minus onset (m:presumed_onset)
per_id = (
    df.groupby("icustayid")
      .agg(max_chart=("m:charttime", "max"), onset=("m:presumed_onset", "first"))
      .reset_index()
)

# Hours since onset; clip negatives to 0
per_id["hours_icu"] = np.maximum((per_id["max_chart"] - per_id["onset"]) / 3600.0, 0.0)

# Merge back
df = df.drop(columns=["hours_icu"], errors="ignore").merge(
    per_id[["icustayid", "hours_icu"]], on="icustayid", how="left"
)

# %% [markdown]
# ### Survivability Stats

# %%
# --- load ---
dfm = pd.read_csv("MIMICtable.csv")

# --- use one row per ICU stay: take the max of each binary label over 4h rows ---
labels = ["died_in_hosp", "died_within_48h_of_out_time", "mortality_90d"]
for c in labels:
    dfm[c] = pd.to_numeric(dfm[c], errors="coerce").fillna(0).clip(0, 1)

by_stay = (dfm.groupby("icustayid")[labels]
             .max()                      # any 1 across rows -> 1 for that stay
             .astype(int)
          )

n = len(by_stay)

def summarize(col, pretty_name):
    deaths = int(by_stay[col].sum())
    mortality_rate = deaths / n
    print(f"{pretty_name}:")
    print(f"  Mortality rate = {mortality_rate:.1%}\n")

summarize("died_within_48h_of_out_time", "48-hours after ICU")
summarize("died_in_hosp", "Hospital mortality")
summarize("mortality_90d", "90-d mortality")


# %% [markdown]
# ### Cohort table

# %%
# JUPYTER CELL — 2) Cohort Table (now using 90-day mortality)  (ROUNDED TO 1 DECIMAL)

def cohort_summary_hosp(table: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, g in table.groupby("morta_inhosp"):
        label = "Survivors" if name == 0 else "Non-survivors"
        pct_f = 100.0 * g["female"].mean(skipna=True) if "female" in g else np.nan
        mean_age = g["age_years"].mean(skipna=True) if "age_years" in g else np.nan
        per_stay = g.drop_duplicates(subset=["m:icustayid"])[["m:icustayid", "hours_icu"]] if "hours_icu" in g else pd.DataFrame(columns=["m:icustayid","hours_icu"])
        mean_hours = per_stay["hours_icu"].median(skipna=True) if not per_stay.empty else np.nan
        n_stays = int(g["m:icustayid"].nunique())
        out.append([label, pct_f, mean_age, mean_hours, n_stays])
    tbl = pd.DataFrame(out, columns=["Group", "% Female", "Mean Age (y)", "Hours in ICU (median)", "Total Population"])
    tbl[["% Female", "Mean Age (y)", "Hours in ICU (median)"]] = (
        tbl[["% Female", "Mean Age (y)", "Hours in ICU (median)"]].astype(float).round(1)
    )
    return tbl

cohort_tbl_hosp = cohort_summary_hosp(df)
cohort_tbl_hosp

# %% [markdown]
# ### AI Clinician recreation

# %%
for c in ["bloc","gender","age","mechvent","max_dose_vaso",
          "died_in_hosp","mortality_90d","died_within_48h_of_out_time"]:
    if c in dfm.columns:
        dfm[c] = pd.to_numeric(dfm[c], errors="coerce")

g = dfm.sort_values(["icustayid","bloc"])

first = g.groupby("icustayid").first()[["gender","age"]]
last_bloc = g.groupby("icustayid")["bloc"].max()
min_bloc  = g.groupby("icustayid")["bloc"].min()

agg = g.groupby("icustayid").agg(
    died_in_hosp=("died_in_hosp","max"),
    mort90=("mortality_90d","max"),
    died48h=("died_within_48h_of_out_time","max"),
    mechvent_any=("mechvent","max"),
    max_dose_vaso_max=("max_dose_vaso","max"),
)

per_stay = first.join([last_bloc.rename("last_bloc"), min_bloc.rename("min_bloc")]).join(agg)

# age to years
per_stay["age_years"] = per_stay["age"] / 365.25

# ICU length (days) from bloc (4h bins)
per_stay["icu_hours"] = np.where(per_stay["min_bloc"].fillna(1) == 0,
                                 (per_stay["last_bloc"] + 1) * 4.0,
                                 per_stay["last_bloc"] * 4.0)
per_stay["icu_days"] = per_stay["icu_hours"] / 24.0

# vasopressor any
per_stay["vaso_any"] = (per_stay["max_dose_vaso_max"] > 0).astype(int)

# male %  (assumes 1=male, 0=female; flip if needed)
N = len(per_stay)
male_n = int((per_stay["gender"] == 1).sum())
male_pct = 100 * male_n / N if N else np.nan

# helpers
def mean_sd(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return (s.mean(), s.std())

def med_iqr(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return (s.median(), s.quantile(0.25), s.quantile(0.75))

def n_pct(mask):
    n = int(mask.sum())
    p = 100 * n / N if N else np.nan
    return n, p

rows = []

# Available counts
rows.append(("Unique ICU admissions (n)", f"{N}", ""))

# Age (mean (sd))
m, s = mean_sd(per_stay["age_years"])
rows.append(("Age, years (mean (s.d.))", f"{m:.1f}", f"{s:.1f}"))

# Male gender (n (%))
rows.append(("Male gender (n (%))", f"{male_n}", f"{male_pct:.1f}%"))

# Initial SOFA (mean (s.d.)) if present at first bloc
if "SOFA" in g.columns:
    sofa_first = g.groupby("icustayid")["SOFA"].first()
    m, s = mean_sd(sofa_first)
    rows.append(("Initial SOFA (mean (s.d.))", f"{m:.1f}", f"{s:.1f}"))
else:
    rows.append(("Initial SOFA (mean (s.d.))", "", ""))

# Procedures during 72 h
mv_n, mv_p = n_pct(per_stay["mechvent_any"] > 0)
rows.append(("Mechanical ventilation (n (%))", f"{mv_n}", f"{mv_p:.1f}%"))

vp_n, vp_p = n_pct(per_stay["vaso_any"] > 0)
rows.append(("Vasopressors (n (%))", f"{vp_n}", f"{vp_p:.1f}%"))

# Length of stay, days (median, IQR)
med, q1, q3 = med_iqr(per_stay["icu_days"])
rows.append(("Length of stay, days (median, IQR)", f"{med:.1f}", f"{q1:.1f}–{q3:.1f}"))

# Death within 48h of ICU out-time
n48, p48 = n_pct(per_stay["died48h"] == 1)
rows.append(("Death within 48h of ICU out-time", f"{p48:.1f}%", f"n={n48}"))

# Hospital mortality
hn, hp = n_pct(per_stay["died_in_hosp"] == 1)
rows.append(("Hospital mortality", f"{hp:.1f}%", f"n={hn}"))

# 90-d mortality
n90, p90 = n_pct(per_stay["mort90"] == 1)
rows.append(("90-d mortality", f"{p90:.1f}%", f"n={n90}"))

table = pd.DataFrame(rows, columns=["Characteristic", "Value", "Bracket/Note"])
print(table.to_string(index=False))

# %% [markdown]
# ### Feature Summary

# %%
# JUPYTER CELL
# Choose feature columns: numeric, excluding ids/timestamps/labels
exclude = {"m:icustayid","survivor","label_in_hosp","sex","female","intime","outtime"}
num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

desc = pd.DataFrame({
    "mean": df[num_cols].mean(),
    "median": df[num_cols].median(),
    "std": df[num_cols].std(),
    "min": df[num_cols].min(),
    "q25": df[num_cols].quantile(0.25),
    "q75": df[num_cols].quantile(0.75),
    "max": df[num_cols].max(),
    "missing_%": 100.0 * (1 - df[num_cols].count()/len(df))
}).T  # descriptors on rows, features on columns
desc


# %% [markdown]
# ### Class Balance and Basic Distributions

# %%
# JUPYTER CELL — 4) Class Balance & Basic Distributions (use inhosp)  (ADJUSTED)
mortality_rate_hosp = dfm["died_in_hosp"].mean()
n_stays = dfm["icustayid"].nunique()
print(f"90-day mortality rate: {mortality_rate_hosp:.3f} (N stays={n_stays})")
by_stay_age = (dfm.groupby("icustayid")["age"]
             .mean()                      # any 1 across rows -> 1 for that stay
             .astype(int) / 365.25
          )
plt.figure()
by_stay_age.dropna().hist(bins=30)
plt.title("Age (years) distribution")
plt.xlabel("Age (years)"); plt.ylabel("Count")
plt.show()

print(dfm.columns)

plt.figure()
df["hours_icu"].dropna().hist(bins=30)
plt.title("ICU stay duration distribution")
plt.xlabel("Hours in ICU"); plt.ylabel("Count")
plt.show()


# %%
import altair as alt
alt.data_transformers.disable_max_rows()

# Metrics
mortality_rate_hosp = dfm["died_in_hosp"].mean()
n_stays = df["icustayid"].nunique()
print(f"In-hospital mortality rate: {mortality_rate_hosp:.3f} (N stays={n_stays})")

# Per-stay aggregates
by_stay = df.groupby("m:icustayid").agg(
    age_val=("o:age", "mean"),          # your 'age' looked like days earlier
    hours_icu=("hours_icu", "median")
).reset_index()
# Convert to years if needed
by_stay["age_years"] = np.where(by_stay["age_val"].max() > 120,
                                by_stay["age_val"] / 365.25,
                                by_stay["age_val"])
# medians
med_age   = float(by_stay['age_years'].median(skipna=True))
med_hours = float(by_stay['hours_icu'].median(skipna=True))

# Age (grey bars) + black median line
age_hist = alt.Chart(by_stay.dropna(subset=['age_years'])).mark_bar(color='#9e9e9e').encode(
    x=alt.X('age_years:Q', bin=alt.Bin(maxbins=30), title='Age (years)'),
    y=alt.Y('count():Q', title='Count')
).properties(title='Age distribution (per stay)')
age_med = alt.Chart(pd.DataFrame({'age_years':[med_age]})).mark_rule(color='black', size=2).encode(x='age_years:Q')
age_chart = (age_hist + age_med).configure_view(strokeWidth=0)

# ICU hours (grey bars) + black median line
icu_hist = alt.Chart(by_stay.dropna(subset=['hours_icu'])).mark_bar(color='#9e9e9e').encode(
    x=alt.X('hours_icu:Q', bin=alt.Bin(maxbins=30), title='Hours in ICU'),
    y=alt.Y('count():Q', title='Count')
).properties(title='ICU length distribution (per stay)')
icu_med = alt.Chart(pd.DataFrame({'hours_icu':[med_hours]})).mark_rule(color='black', size=2).encode(x='hours_icu:Q')
icu_chart = (icu_hist + icu_med).configure_view(strokeWidth=0)

age_chart

# %%
icu_chart

# %% [markdown]
# ### Missingness Overview

# %%
# JUPYTER CELL
missing_per_feature = df[num_cols].isna().mean().sort_values(ascending=False)
missing_df = missing_per_feature.rename("missing_%").mul(100).to_frame()
missing_df.head(20)


# %%
# JUPYTER CELL
# Bar chart of missingness (single plot)
plt.figure(figsize=(10,4))
missing_per_feature.mul(100).plot(kind="bar")
plt.title("Missingness per numeric feature (%)")
plt.xlabel("Feature"); plt.ylabel("% Missing")
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Correlation Matrix

# %%
# JUPYTER CELL
corr = df[num_cols].corr()
plt.figure(figsize=(6,5))
plt.imshow(corr.values, aspect='auto')
plt.colorbar()
plt.title("Correlation matrix (numeric features)")
plt.xticks(range(len(num_cols)), num_cols, rotation=90)
plt.yticks(range(len(num_cols)), num_cols)
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Actions Distributions

# %%
# JUPYTER CELL
# Choose columns (adjust to your names if needed)
iv_col  = "o:input_4hourly"  if "o:input_4hourly" in df.columns else None
vp_col  = "o:max_dose_vaso" if "o:max_dose_vaso" in df.columns else None
assert iv_col is not None and vp_col is not None, "IV or VP columns not found—set COLUMN_MAP."

iv = pd.to_numeric(df[iv_col], errors="coerce").fillna(0.0)
vp = pd.to_numeric(df[vp_col], errors="coerce").fillna(0.0)

# Raw 2D histogram (single plot)
H, xedges, yedges = np.histogram2d(iv.values, vp.values, bins=40)
plt.figure()
plt.imshow(H.T, origin='lower', aspect='auto')
plt.title("2D histogram: IV vs VP (raw dosages)")
plt.xlabel("IV"); plt.ylabel("VP")
plt.colorbar()
plt.show()

# 5x5 action bins (bin 0 for zero; bins 1..4 by quartiles of non-zero)
def make_quartile_bins(x):
    x = pd.to_numeric(x, errors="coerce").fillna(0.0)
    nz = x[x > 0]
    if len(nz) == 0:
        cuts = [np.inf, np.inf, np.inf]
    else:
        cuts = nz.quantile([0.25, 0.5, 0.75]).values.tolist()
    # 0 if zero; else 1..4 by thresholds
    def assign(v):
        if v <= 0: return 0
        return 1 + sum(v > t for t in cuts)
    return x.apply(assign).astype(int)

iv_bin = make_quartile_bins(df[iv_col])
vp_bin = make_quartile_bins(df[vp_col])

# Count heatmap 5x5
counts = pd.crosstab(iv_bin, vp_bin).reindex(index=range(5), columns=range(5), fill_value=0)
plt.figure()
plt.imshow(counts.values, origin='lower')
plt.title("Action counts (IV bin x VP bin) [5x5]")
plt.xlabel("VP bin (0..4)"); plt.ylabel("IV bin (0..4)")
plt.colorbar()
plt.show()



