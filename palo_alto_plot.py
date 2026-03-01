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
