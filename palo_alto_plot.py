"""
Filter sunset data near Palo Alto (37.4°N, 122.14°W) and plot
local sunset time vs day of year.
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv("sunset.csv")
df = df.dropna(subset=["sunset_time"]).copy()

df["sunset_minutes"] = df["sunset_time"].apply(
    lambda t: int(t.split(":")[0]) * 60 + int(t.split(":")[1])
)
df["date"] = pd.to_datetime(df["date"])
df["day_of_year"] = df["date"].dt.dayofyear

# ── Filter near Palo Alto ─────────────────────────────────────────────────────
# Palo Alto: 37.4°N, -122.14°W
# Data is on a 5° grid, so nearest points are lat=35/40, lon=-120/-125

PA_LAT, PA_LON = 37.4, -122.14
RADIUS = 5.0  # degrees

nearby = df[
    (df["latitude"].between(PA_LAT - RADIUS, PA_LAT + RADIUS))
    & (df["longitude"].between(PA_LON - RADIUS, PA_LON + RADIUS))
].copy()

print(
    f"Palo Alto area: lat [{PA_LAT-RADIUS}, {PA_LAT+RADIUS}], "
    f"lon [{PA_LON-RADIUS}, {PA_LON+RADIUS}]"
)
print(f"Matching rows: {len(nearby)}")
print(f"Unique (lat, lon) pairs:")
for (lat, lon), cnt in nearby.groupby(["latitude", "longitude"]).size().items():
    dist = np.sqrt((lat - PA_LAT) ** 2 + (lon - PA_LON) ** 2)
    print(f"  ({lat:+6.1f}, {lon:+7.1f})  —  {cnt} rows  (dist={dist:.1f}°)")

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))

# Color by (lat, lon) pair
pairs = nearby.groupby(["latitude", "longitude"])
cmap = plt.cm.get_cmap("tab10", len(pairs))

for i, ((lat, lon), grp) in enumerate(pairs):
    grp_sorted = grp.sort_values("day_of_year")
    ax.plot(
        grp_sorted["day_of_year"],
        grp_sorted["sunset_minutes"],
        "o-",
        color=cmap(i),
        markersize=3,
        linewidth=1,
        label=f"({lat:.0f}°, {lon:.0f}°)",
        alpha=0.8,
    )

ax.set_xlabel("Day of Year", fontsize=12)
ax.set_ylabel("Local Sunset Time", fontsize=12)
ax.set_title("Local Sunset Time vs Day of Year — Near Palo Alto", fontsize=14)

# y-axis as HH:MM
yticks = range(900, 1321, 60)
ax.set_yticks(list(yticks))
ax.set_yticklabels([f"{t//60}:{t%60:02d}" for t in yticks])
ax.set_xlim(0, 370)
ax.grid(True, alpha=0.3)
ax.legend(title="(lat, lon)", fontsize=9)

plt.tight_layout()
plt.savefig("palo_alto_sunset.png", dpi=150)
plt.close()
print("\n✓ Saved → palo_alto_sunset.png")

# ── Train model on nearby data ────────────────────────────────────────────────

from sklearn.linear_model import LinearRegression

# Features: day_sin, day_cos, latitude, tz_offset
nearby["day_sin"] = np.sin(2 * np.pi * nearby["day_of_year"] / 365.25)
nearby["day_cos"] = np.cos(2 * np.pi * nearby["day_of_year"] / 365.25)
nearby["tz_offset"] = ((nearby["longitude"] + 7.5) % 15) - 7.5
nearby["lat_x_day_sin"] = nearby["latitude"] * nearby["day_sin"]
nearby["lat_x_day_cos"] = nearby["latitude"] * nearby["day_cos"]

feature_cols = [
    "latitude",
    "tz_offset",
    "day_sin",
    "day_cos",
    "lat_x_day_sin",
    "lat_x_day_cos",
]

X = nearby[feature_cols].values
y = nearby["sunset_minutes"].values

lr = LinearRegression()
lr.fit(X, y)

from sklearn.metrics import mean_absolute_error, r2_score

y_pred = lr.predict(X)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\n" + "=" * 60)
print("MODEL (trained on Palo Alto nearby data only)")
print("=" * 60)
print(f"  Training rows: {len(nearby)}")
print(f"  MAE  = {mae:.2f} min")
print(f"  R²   = {r2:.4f}")
print(f"\n  Coefficients:")
for name, coef in zip(feature_cols, lr.coef_):
    print(f"    {name:20s} : {coef:+.4f}")
print(f"    {'intercept':20s} : {lr.intercept_:+.4f}")

# ── Predict Palo Alto, Feb 28 ────────────────────────────────────────────────

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
        ]
    ]
)

pred = lr.predict(pa_x)[0]
h, m = int(pred // 60), int(pred % 60)

print(f"\n{'=' * 60}")
print(f"PREDICTION: Palo Alto (37.4°N, 122.14°W), Feb 28")
print(f"{'=' * 60}")
print(f"  Predicted sunset: {h}:{m:02d}  ({pred:.0f} min)")
print()
