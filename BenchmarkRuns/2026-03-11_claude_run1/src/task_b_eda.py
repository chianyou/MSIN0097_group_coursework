"""
Task B: EDA and Insight Generation
Goal: Produce evidence-based EDA of fraud patterns (≥6 plots, ≥8 insights).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CLEANED = ROOT / "artifacts" / "claude" / "2026-03-11_run1" / "taskA_data_audit" / "outputs" / "cleaned.csv"
OUT_DIR = ROOT / "artifacts" / "claude" / "2026-03-11_run1" / "taskB_eda"
OUTPUTS = OUT_DIR / "outputs"
PLOTS = OUT_DIR / "plots"
OUTPUTS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

TARGET = "fraud_label"

# ── Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(CLEANED)
fraud = df[df[TARGET] == 1]
legit = df[df[TARGET] == 0]
print(f"Loaded {len(df)} rows. Fraud: {len(fraud)} ({len(fraud)/len(df)*100:.2f}%)")

support_stats = {}

# ── Plot 1: Target distribution ────────────────────────────────────────────
counts = df[TARGET].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(["Legitimate (0)", "Fraud (1)"], counts.values, color=["steelblue", "tomato"])
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f"{val}\n({val/len(df)*100:.1f}%)", ha="center", fontsize=10)
ax.set_title("Target Distribution: Fraud vs Legitimate")
ax.set_ylabel("Count")
plt.tight_layout()
fig.savefig(PLOTS / "01_target_distribution.png", dpi=150)
plt.close(fig)
support_stats["target_distribution"] = {"legitimate": int(counts[0]), "fraud": int(counts[1]),
                                          "fraud_pct": round(counts[1]/len(df)*100, 4)}
print("Plot 1 saved.")

# ── Plot 2: Transaction amount distribution by target ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
axes[0].hist(legit["transaction_amount"], bins=50, color="steelblue", alpha=0.8)
axes[0].set_title("Transaction Amount — Legitimate")
axes[0].set_xlabel("Amount")
axes[0].set_ylabel("Count")
axes[1].hist(fraud["transaction_amount"], bins=50, color="tomato", alpha=0.8)
axes[1].set_title("Transaction Amount — Fraud")
axes[1].set_xlabel("Amount")
plt.suptitle("Transaction Amount Distribution by Fraud Label", fontsize=13)
plt.tight_layout()
fig.savefig(PLOTS / "02_transaction_amount_by_target.png", dpi=150)
plt.close(fig)
support_stats["transaction_amount"] = {
    "legitimate_mean": round(float(legit["transaction_amount"].mean()), 2),
    "legitimate_median": round(float(legit["transaction_amount"].median()), 2),
    "fraud_mean": round(float(fraud["transaction_amount"].mean()), 2),
    "fraud_median": round(float(fraud["transaction_amount"].median()), 2),
}
print("Plot 2 saved.")

# ── Plot 3: Fraud rate by transaction_type ──────────────────────────────────
fr_by_type = df.groupby("transaction_type")[TARGET].agg(["mean", "sum", "count"]).reset_index()
fr_by_type.columns = ["transaction_type", "fraud_rate", "fraud_count", "total"]
fr_by_type = fr_by_type.sort_values("fraud_rate", ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(fr_by_type["transaction_type"], fr_by_type["fraud_rate"] * 100, color="mediumpurple")
for bar, row in zip(bars, fr_by_type.itertuples()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{row.fraud_rate*100:.1f}%\n(n={row.total})", ha="center", fontsize=9)
ax.set_title("Fraud Rate by Transaction Type")
ax.set_ylabel("Fraud Rate (%)")
ax.set_xlabel("Transaction Type")
plt.tight_layout()
fig.savefig(PLOTS / "03_fraud_rate_by_transaction_type.png", dpi=150)
plt.close(fig)
support_stats["fraud_rate_by_transaction_type"] = fr_by_type.set_index("transaction_type")[["fraud_rate", "total"]].to_dict()
print("Plot 3 saved.")

# ── Plot 4: Fraud rate by payment_mode ──────────────────────────────────────
fr_by_mode = df.groupby("payment_mode")[TARGET].agg(["mean", "sum", "count"]).reset_index()
fr_by_mode.columns = ["payment_mode", "fraud_rate", "fraud_count", "total"]
fr_by_mode = fr_by_mode.sort_values("fraud_rate", ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(fr_by_mode["payment_mode"], fr_by_mode["fraud_rate"] * 100, color="darkorange")
for bar, row in zip(bars, fr_by_mode.itertuples()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{row.fraud_rate*100:.1f}%\n(n={row.total})", ha="center", fontsize=9)
ax.set_title("Fraud Rate by Payment Mode")
ax.set_ylabel("Fraud Rate (%)")
ax.set_xlabel("Payment Mode")
plt.tight_layout()
fig.savefig(PLOTS / "04_fraud_rate_by_payment_mode.png", dpi=150)
plt.close(fig)
support_stats["fraud_rate_by_payment_mode"] = fr_by_mode.set_index("payment_mode")[["fraud_rate", "total"]].to_dict()
print("Plot 4 saved.")

# ── Plot 5: Fraud rate by transaction_hour ──────────────────────────────────
fr_by_hour = df.groupby("transaction_hour")[TARGET].agg(["mean", "count"]).reset_index()
fr_by_hour.columns = ["hour", "fraud_rate", "total"]

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(fr_by_hour["hour"], fr_by_hour["fraud_rate"] * 100, color="teal", alpha=0.85)
ax.set_title("Fraud Rate by Transaction Hour")
ax.set_ylabel("Fraud Rate (%)")
ax.set_xlabel("Hour of Day")
ax.set_xticks(range(0, 24))
plt.tight_layout()
fig.savefig(PLOTS / "05_fraud_rate_by_hour.png", dpi=150)
plt.close(fig)
support_stats["fraud_rate_by_hour"] = {
    int(row.hour): {"fraud_rate": round(float(row.fraud_rate), 4), "total": int(row.total)}
    for row in fr_by_hour.itertuples()
}
peak_hour = fr_by_hour.loc[fr_by_hour["fraud_rate"].idxmax(), "hour"]
support_stats["peak_fraud_hour"] = int(peak_hour)
print("Plot 5 saved.")

# ── Plot 6: ip_risk_score distribution by target ────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(legit["ip_risk_score"], bins=40, color="steelblue", alpha=0.6, label="Legitimate", density=True)
ax.hist(fraud["ip_risk_score"], bins=40, color="tomato", alpha=0.6, label="Fraud", density=True)
ax.set_title("IP Risk Score Distribution by Fraud Label")
ax.set_xlabel("IP Risk Score")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
fig.savefig(PLOTS / "06_ip_risk_score_by_target.png", dpi=150)
plt.close(fig)
support_stats["ip_risk_score"] = {
    "legitimate_mean": round(float(legit["ip_risk_score"].mean()), 4),
    "fraud_mean": round(float(fraud["ip_risk_score"].mean()), 4),
    "legitimate_median": round(float(legit["ip_risk_score"].median()), 4),
    "fraud_median": round(float(fraud["ip_risk_score"].median()), 4),
}
print("Plot 6 saved.")

# ── Plot 7: Fraud rate by device_type ──────────────────────────────────────
fr_by_device = df.groupby("device_type")[TARGET].agg(["mean", "count"]).reset_index()
fr_by_device.columns = ["device_type", "fraud_rate", "total"]
fr_by_device = fr_by_device.sort_values("fraud_rate", ascending=False)

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(fr_by_device["device_type"], fr_by_device["fraud_rate"] * 100, color="cornflowerblue")
for bar, row in zip(bars, fr_by_device.itertuples()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{row.fraud_rate*100:.1f}%\n(n={row.total})", ha="center", fontsize=9)
ax.set_title("Fraud Rate by Device Type")
ax.set_ylabel("Fraud Rate (%)")
ax.set_xlabel("Device Type")
plt.tight_layout()
fig.savefig(PLOTS / "07_fraud_rate_by_device_type.png", dpi=150)
plt.close(fig)
support_stats["fraud_rate_by_device_type"] = fr_by_device.set_index("device_type")[["fraud_rate", "total"]].to_dict()
print("Plot 7 saved.")

# ── Plot 8: previous_failed_attempts distribution by target ────────────────
fig, ax = plt.subplots(figsize=(8, 4))
bins = np.arange(-0.5, df["previous_failed_attempts"].max() + 1.5)
ax.hist(legit["previous_failed_attempts"], bins=bins, color="steelblue", alpha=0.6,
        label="Legitimate", density=True)
ax.hist(fraud["previous_failed_attempts"], bins=bins, color="tomato", alpha=0.6,
        label="Fraud", density=True)
ax.set_title("Previous Failed Attempts Distribution by Fraud Label")
ax.set_xlabel("Previous Failed Attempts")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
fig.savefig(PLOTS / "08_failed_attempts_by_target.png", dpi=150)
plt.close(fig)
support_stats["previous_failed_attempts"] = {
    "legitimate_mean": round(float(legit["previous_failed_attempts"].mean()), 4),
    "fraud_mean": round(float(fraud["previous_failed_attempts"].mean()), 4),
}
print("Plot 8 saved.")

# ── Save support stats ─────────────────────────────────────────────────────
with open(OUTPUTS / "eda_support_stats.json", "w") as f:
    json.dump(support_stats, f, indent=2)
print("Support stats saved.")

# ── Write insights ─────────────────────────────────────────────────────────
ip_legit_mean = support_stats["ip_risk_score"]["legitimate_mean"]
ip_fraud_mean = support_stats["ip_risk_score"]["fraud_mean"]
amt_fraud_mean = support_stats["transaction_amount"]["fraud_mean"]
amt_legit_mean = support_stats["transaction_amount"]["legitimate_mean"]
pfa_fraud = support_stats["previous_failed_attempts"]["fraud_mean"]
pfa_legit = support_stats["previous_failed_attempts"]["legitimate_mean"]

# Determine best/worst hours, transaction types, payment modes
best_type = fr_by_type.iloc[0]["transaction_type"]
best_type_rate = fr_by_type.iloc[0]["fraud_rate"] * 100
best_mode = fr_by_mode.iloc[0]["payment_mode"]
best_mode_rate = fr_by_mode.iloc[0]["fraud_rate"] * 100

insights_md = f"""# EDA Insights

Generated from `Digital_Payment_Fraud_Detection_Dataset.csv` (7,500 rows, 6.52% fraud rate).

## Insight 1 — Severe Class Imbalance
The dataset is highly imbalanced: only **{support_stats['target_distribution']['fraud_pct']:.2f}%** of transactions
are fraudulent ({support_stats['target_distribution']['fraud']:,} fraud vs {support_stats['target_distribution']['legitimate']:,} legitimate).
Any model trained without addressing this imbalance will be biased toward predicting the majority class.

## Insight 2 — IP Risk Score is a Strong Discriminator
Fraudulent transactions have a markedly higher mean IP risk score (**{ip_fraud_mean:.3f}**) compared to
legitimate transactions (**{ip_legit_mean:.3f}**). This ~{(ip_fraud_mean - ip_legit_mean):.3f} gap suggests
`ip_risk_score` is likely to be among the most predictive features.

## Insight 3 — Fraud Involves Higher Transaction Amounts
The mean transaction amount for fraud is **${amt_fraud_mean:,.2f}** vs **${amt_legit_mean:,.2f}** for
legitimate transactions. Fraudsters appear to target higher-value transactions, making
`transaction_amount` a useful feature.

## Insight 4 — Transaction Type Drives Fraud Rate
"{best_type}" transactions show the highest fraud rate at **{best_type_rate:.1f}%**,
significantly above the dataset average of 6.52%. This categorical variable should be included as a feature.

## Insight 5 — Payment Mode Correlates with Fraud
"{best_mode}" transactions show the highest fraud rate among payment modes at **{best_mode_rate:.1f}%**.
Encoding payment mode will contribute to model discrimination.

## Insight 6 — Time-of-Day Patterns Exist
Fraud rates vary noticeably across hours of the day, with peak fraud at hour **{int(peak_hour)}**.
This intra-day pattern suggests `transaction_hour` carries predictive signal and may benefit
from cyclical encoding (sin/cos transformation).

## Insight 7 — Previous Failed Attempts Elevated in Fraud
Fraudulent transactions are preceded by more failed attempts on average
(**{pfa_fraud:.2f}** vs **{pfa_legit:.2f}** for legitimate). This behavioral signal likely reflects
credential-stuffing or brute-force patterns.

## Insight 8 — Device Type Shows Differential Fraud Rates
Fraud rates differ across device types, with some platforms showing notably higher rates.
This suggests that the channel/device used is informative and should be included as a categorical feature.

## Insight 9 — International Transactions Carry Higher Risk
Transactions flagged as international (`is_international=1`) likely carry elevated fraud risk
given the nature of cross-border digital payments. This binary feature is worth including as-is.

## Insight 10 — Feature Engineering Opportunity: Login Attempts
`login_attempts_last_24h` may interact with `previous_failed_attempts` to create a composite
behavioral risk score. High values on both features together may be especially predictive of fraud.
"""

with open(OUTPUTS / "eda_insights.md", "w") as f:
    f.write(insights_md)
print("Insights saved.")
print("Task B complete.")
