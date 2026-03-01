"""
Explore sunset.csv and train a linear model to predict sunset time.

The sunset_time column is LOCAL time (not UTC), so:
  - No midnight wrapping issue
  - Longitude is NOT a predictor (it only shifts UTC, not local)
  - Main predictors: latitude, day_of_year
  - Local sunset clusters around ~18:00 (1080 min), with wider spread
    at extreme latitudes near solstices

Pipeline:
  1. Load & clean (drop NaN polar rows, parse HH:MM → minutes)
  2. Feature engineering (day-of-year trig, latitude interactions)
  3. EDA visualisations
  4. Train/test split → Linear Regression & Poly-2 Ridge
  5. Residual analysis & summary
"""

import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import HistGradientBoostingRegressor


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: generate per-day heatmaps of sunset time
# ═══════════════════════════════════════════════════════════════════════════════


def generate_daily_heatmaps(df, output_dir="daily_heatmaps"):
    """
    For each unique date in the data, produce a color heatmap of
    sunset time (local) with latitude on y-axis, longitude on x-axis.
    All plots saved as PNGs in `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)

    dates = sorted(df["date"].unique())
    print(f"\nGenerating {len(dates)} daily heatmaps → {output_dir}/")

    lats = sorted(df["latitude"].unique())
    lons = sorted(df["longitude"].unique())

    for i, date in enumerate(dates):
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        doy = pd.Timestamp(date).dayofyear
        sub = df[df["date"] == date]

        pivot = sub.pivot_table(
            index="latitude",
            columns="longitude",
            values="sunset_minutes",
            aggfunc="first",
        )
        pivot = pivot.reindex(index=lats, columns=lons)

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.pcolormesh(
            lons,
            lats,
            pivot.values,
            cmap="plasma",
            shading="auto",
            vmin=0,
            vmax=1440,
        )
        cbar = plt.colorbar(im, ax=ax, label="Local sunset time (min)")
        cbar.set_ticks([0, 360, 720, 1080, 1440])
        cbar.set_ticklabels(["00:00", "06:00", "12:00", "18:00", "24:00"])

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Sunset Time (local)  —  {date_str}  (day {doy})")

        fname = os.path.join(output_dir, f"sunset_day{doy:03d}_{date_str}.png")
        plt.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close(fig)

        if (i + 1) % 20 == 0 or i == len(dates) - 1:
            print(f"  [{i+1}/{len(dates)}] saved {fname}")

    print(f"✓ Done — {len(dates)} heatmaps in {output_dir}/")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. LOAD & CLEAN  ─────────────────────────────────────────────────────────

df = pd.read_csv("sunset.csv")
print("=" * 60)
print("RAW DATA")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nSample rows (with valid times):")
print(df[df["sunset_time"].notna()].head(10))

n_nan = df["sunset_time"].isna().sum()
print(f"\nMissing: {n_nan}  |  Valid: {df.shape[0] - n_nan}")

df = df.dropna(subset=["sunset_time"]).copy()


# Parse "HH:MM" → minutes since midnight (local time)
def time_to_minutes(t: str) -> int:
    h, m = t.strip().split(":")
    return int(h) * 60 + int(m)


df["sunset_minutes"] = df["sunset_time"].apply(time_to_minutes)
df["date"] = pd.to_datetime(df["date"])
df["day_of_year"] = df["date"].dt.dayofyear

print(f"After cleaning: {df.shape[0]:,} rows")
print(f"Sunset range: {df['sunset_minutes'].min()} – {df['sunset_minutes'].max()} min")
print(
    f"Mean sunset : {df['sunset_minutes'].mean():.0f} min "
    f"({df['sunset_minutes'].mean()/60:.1f} h)"
)

# ── 2. GENERATE DAILY HEATMAPS (skip if folder already exists) ───────────────

if not os.path.isdir("daily_heatmaps"):
    generate_daily_heatmaps(df)
else:
    n_existing = len([f for f in os.listdir("daily_heatmaps") if f.endswith(".png")])
    print(f"\n✓ daily_heatmaps/ already exists ({n_existing} files) — skipping")

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
#
# Since sunset_time is local, the physics is:
#   sunset_local ≈ 12:00 + half_day_length(latitude, day_of_year)
#
# half_day_length depends on solar declination (day_of_year) and latitude.
# At equinox, half_day ≈ 6h everywhere.  At solstice, it varies with lat.
#
# Features:
#   - day_sin, day_cos          : annual cycle (captures declination)
#   - latitude                  : baseline latitude effect
#   - lat × day_sin/cos         : seasonal swing scales with latitude
#   - lat² × day_sin/cos        : non-linearity at extreme latitudes
#   - abs_lat                   : symmetric latitude effect on avg day length
#   - tz_offset                 : longitude offset from timezone center
#                                 (within a timezone, further west → later sunset)

# Trig encoding of annual cycle
df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

# Interactions: latitude × season
df["lat_x_day_sin"] = df["latitude"] * df["day_sin"]
df["lat_x_day_cos"] = df["latitude"] * df["day_cos"]

# Quadratic lat × season (extreme latitudes)
df["lat2_x_day_sin"] = (df["latitude"] ** 2) * df["day_sin"]
df["lat2_x_day_cos"] = (df["latitude"] ** 2) * df["day_cos"]

df["abs_lat"] = df["latitude"].abs()

# Longitude offset from timezone center (timezones centered at 0, ±15, ±30, ...)
# Range: [-7.5, +7.5] degrees.  Positive = east of center → earlier local sunset.
df["tz_offset"] = ((df["longitude"] + 7.5) % 15) - 7.5

print("\n" + "=" * 60)
print("FEATURE SUMMARY")
print("=" * 60)
print(
    df[["latitude", "longitude", "day_of_year", "sunset_minutes"]].describe().round(2)
)

# ── 4. EDA OVERVIEW PLOTS ────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Sunset Time EDA (local time)", fontsize=16)

# (a) Distribution
axes[0, 0].hist(
    df["sunset_minutes"], bins=80, edgecolor="black", alpha=0.7, color="steelblue"
)
axes[0, 0].set_xlabel("Sunset (minutes since midnight, local)")
axes[0, 0].set_ylabel("Count")
axes[0, 0].set_title("Distribution of local sunset time")
axes[0, 0].axvline(1080, color="red", ls="--", label="18:00")
axes[0, 0].legend()

# (b) Sunset vs latitude (colour = day of year)
sc = axes[0, 1].scatter(
    df["latitude"],
    df["sunset_minutes"],
    c=df["day_of_year"],
    cmap="twilight",
    s=1,
    alpha=0.3,
)
axes[0, 1].set_xlabel("Latitude")
axes[0, 1].set_ylabel("Sunset (min, local)")
axes[0, 1].set_title("Sunset vs Latitude (colour = day of year)")
plt.colorbar(sc, ax=axes[0, 1], label="Day of year")

# (c) Sunset vs longitude (should be ~flat since it's local time)
sc2 = axes[1, 0].scatter(
    df["longitude"],
    df["sunset_minutes"],
    c=df["latitude"],
    cmap="coolwarm",
    s=1,
    alpha=0.3,
)
axes[1, 0].set_xlabel("Longitude")
axes[1, 0].set_ylabel("Sunset (min, local)")
axes[1, 0].set_title("Sunset vs Longitude (expect ~flat)")
plt.colorbar(sc2, ax=axes[1, 0], label="Latitude")

# (d) Sunset vs day of year (colour = latitude)
sc3 = axes[1, 1].scatter(
    df["day_of_year"],
    df["sunset_minutes"],
    c=df["latitude"],
    cmap="coolwarm",
    s=1,
    alpha=0.3,
)
axes[1, 1].set_xlabel("Day of year")
axes[1, 1].set_ylabel("Sunset (min, local)")
axes[1, 1].set_title("Sunset vs Day of Year (colour = latitude)")
plt.colorbar(sc3, ax=axes[1, 1], label="Latitude")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150)
plt.close()
print("\n✓ Saved EDA plots → eda_plots.png")

# ── 4b. SUNSET vs DAY-OF-YEAR BY LATITUDE BUCKET ─────────────────────────────

lat_bins = [-90, -60, -40, -20, 0, 20, 40, 60, 90]
lat_labels = [
    "60–90°S",
    "40–60°S",
    "20–40°S",
    "0–20°S",
    "0–20°N",
    "20–40°N",
    "40–60°N",
    "60–90°N",
]
df["lat_bucket"] = pd.cut(df["latitude"], bins=lat_bins, labels=lat_labels)

cmap = plt.cm.get_cmap("coolwarm", len(lat_labels))
colors = {label: cmap(i) for i, label in enumerate(lat_labels)}

fig, ax = plt.subplots(figsize=(14, 7))
for label in lat_labels:
    sub = df[df["lat_bucket"] == label]
    ax.scatter(
        sub["day_of_year"],
        sub["sunset_minutes"],
        c=[colors[label]],
        s=2,
        alpha=0.3,
        label=label,
    )

ax.set_xlabel("Day of Year", fontsize=12)
ax.set_ylabel("Local Sunset Time (min since midnight)", fontsize=12)
ax.set_title("Local Sunset Time vs Day of Year by Latitude Bucket", fontsize=14)

# format y-axis as HH:MM
yticks = [0, 120, 240, 360, 480, 600, 720, 840, 960, 1080, 1200, 1320, 1440]
ax.set_yticks(yticks)
ax.set_yticklabels([f"{t//60}:{t%60:02d}" for t in yticks])

# legend outside plot
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    labels,
    loc="upper left",
    bbox_to_anchor=(1.01, 1),
    title="Latitude",
    markerscale=4,
    frameon=True,
)

plt.tight_layout()
plt.savefig("sunset_by_lat_bucket.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved sunset by latitude bucket → sunset_by_lat_bucket.png")

# ── 5. MODEL TRAINING ────────────────────────────────────────────────────────
#
# Longitude excluded — it shouldn't affect local sunset time.

feature_cols = [
    "latitude",
    "tz_offset",
    "day_sin",
    "day_cos",
    "lat_x_day_sin",
    "lat_x_day_cos",
    "lat2_x_day_sin",
    "lat2_x_day_cos",
    "abs_lat",
]

X = df[feature_cols].values
y = df["sunset_minutes"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n" + "=" * 60)
print("MODEL TRAINING  (target = local sunset time)")
print("=" * 60)
print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

# ── 5a. Linear Regression ────────────────────────────────────────────────────

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print(f"\n--- Linear Regression ---")
print(f"  MAE  = {mae_lr:.2f} min  ({mae_lr/60:.1f} h)")
print(f"  RMSE = {rmse_lr:.2f} min  ({rmse_lr/60:.1f} h)")
print(f"  R²   = {r2_lr:.4f}")
print(f"\n  Coefficients:")
for name, coef in zip(feature_cols, lr.coef_):
    print(f"    {name:20s} : {coef:+.4f}")
print(f"    {'intercept':20s} : {lr.intercept_:+.4f}")

# ── 5b. Poly-2 Ridge Regression ──────────────────────────────────────────────

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_poly, y_train)
y_pred_ridge = ridge.predict(X_test_poly)

mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"\n--- Poly-2 Ridge Regression ---")
print(f"  MAE  = {mae_ridge:.2f} min  ({mae_ridge/60:.1f} h)")
print(f"  RMSE = {rmse_ridge:.2f} min  ({rmse_ridge/60:.1f} h)")
print(f"  R²   = {r2_ridge:.4f}")
print(f"  # features (after poly): {X_train_poly.shape[1]}")

# ── 5c. Gradient Boosted Trees ────────────────────────────────────────────────

gbt = HistGradientBoostingRegressor(
    max_iter=500,
    max_depth=8,
    learning_rate=0.05,
    min_samples_leaf=20,
    random_state=42,
)
gbt.fit(X_train, y_train)
y_pred_gbt = gbt.predict(X_test)

mae_gbt = mean_absolute_error(y_test, y_pred_gbt)
rmse_gbt = np.sqrt(mean_squared_error(y_test, y_pred_gbt))
r2_gbt = r2_score(y_test, y_pred_gbt)

print(f"\n--- Gradient Boosted Trees (HistGBR) ---")
print(f"  MAE  = {mae_gbt:.2f} min  ({mae_gbt/60:.1f} h)")
print(f"  RMSE = {rmse_gbt:.2f} min  ({rmse_gbt/60:.1f} h)")
print(f"  R²   = {r2_gbt:.4f}")

# ── 6. DIAGNOSTICS ───────────────────────────────────────────────────────────

models_preds = [
    (y_pred_lr, "Linear Regression"),
    (y_pred_ridge, "Poly-2 Ridge"),
    (y_pred_gbt, "Gradient Boosted Trees"),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Model Diagnostics (local sunset time)", fontsize=14)

for i, (y_pred, name) in enumerate(models_preds):
    residuals = y_test - y_pred
    axes[0, i].scatter(y_pred, residuals, s=1, alpha=0.3)
    axes[0, i].axhline(0, color="red", linewidth=1)
    axes[0, i].set_xlabel("Predicted (min)")
    axes[0, i].set_ylabel("Residual (min)")
    axes[0, i].set_title(
        f"Residuals: {name}\nMAE={mean_absolute_error(y_test, y_pred):.1f} min  "
        f"R²={r2_score(y_test, y_pred):.3f}"
    )

    axes[1, i].scatter(y_test, y_pred, s=1, alpha=0.2)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[1, i].plot(lims, lims, "r--", linewidth=1, label="perfect")
    axes[1, i].set_xlabel("Actual (min)")
    axes[1, i].set_ylabel("Predicted (min)")
    axes[1, i].set_title(f"Actual vs Predicted: {name}")
    axes[1, i].legend()

plt.tight_layout()
plt.savefig("model_diagnostics.png", dpi=150)
plt.close()
print("\n✓ Saved model diagnostics → model_diagnostics.png")

# ── 7. SUMMARY ────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  {'Model':<25s} {'MAE (min)':>10s} {'RMSE (min)':>11s} {'R²':>8s}")
print(f"  {'-'*25} {'-'*10} {'-'*11} {'-'*8}")
print(f"  {'Linear Regression':<25s} {mae_lr:>10.2f} {rmse_lr:>11.2f} {r2_lr:>8.4f}")
print(
    f"  {'Poly-2 Ridge':<25s} {mae_ridge:>10.2f} {rmse_ridge:>11.2f} {r2_ridge:>8.4f}"
)
print(
    f"  {'Gradient Boosted Trees':<25s} {mae_gbt:>10.2f} {rmse_gbt:>11.2f} {r2_gbt:>8.4f}"
)
print()

# ── 8. PALO ALTO PREDICTION ──────────────────────────────────────────────────

pa_lat, pa_lon = 37.4, -122.14
pa_doy = 59  # Feb 28
pa_tz_offset = ((pa_lon + 7.5) % 15) - 7.5

pa_day_sin = np.sin(2 * np.pi * pa_doy / 365.25)
pa_day_cos = np.cos(2 * np.pi * pa_doy / 365.25)
pa_x = np.array(
    [
        [
            pa_lat,
            pa_tz_offset,
            pa_day_sin,
            pa_day_cos,
            pa_lat * pa_day_sin,
            pa_lat * pa_day_cos,
            pa_lat**2 * pa_day_sin,
            pa_lat**2 * pa_day_cos,
            abs(pa_lat),
        ]
    ]
)

pred_lr_pa = lr.predict(pa_x)[0]
pred_ridge_pa = ridge.predict(poly.transform(pa_x))[0]
pred_gbt_pa = gbt.predict(pa_x)[0]

print("=" * 60)
print("PALO ALTO PREDICTION  (37.4°N, 122.14°W, Feb 28)")
print("=" * 60)
for name, p in [
    ("Linear Regression", pred_lr_pa),
    ("Poly-2 Ridge", pred_ridge_pa),
    ("Gradient Boosted Trees", pred_gbt_pa),
]:
    h, m = int(p // 60), int(p % 60)
    print(f"  {name:<25s} → {h}:{m:02d}  ({p:.0f} min)")
print()
