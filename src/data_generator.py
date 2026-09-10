"""
PowerGuard AI - Sample Electricity Billing Dataset Generator
Generates realistic electricity billing data with embedded fraud patterns.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

random.seed(42)
np.random.seed(42)

# ─── Config ────────────────────────────────────────────────────────────────────
NUM_CONSUMERS = 80
MONTHS = 12          # 12 billing months
FRAUD_FRACTION = 0.20  # ~20% of consumers have fraud patterns

CONSUMER_TYPES = {
    "residential_low":    {"base": 150, "std": 30,  "weight": 0.35},
    "residential_medium": {"base": 280, "std": 50,  "weight": 0.30},
    "residential_high":   {"base": 500, "std": 80,  "weight": 0.15},
    "commercial_small":   {"base": 700, "std": 100, "weight": 0.12},
    "commercial_large":   {"base": 1500,"std": 200, "weight": 0.08},
}

RATE_PER_UNIT = 6.50   # INR per kWh (illustrative)

FRAUD_TYPES = [
    "sudden_drop",        # meter tampering / bypass
    "reading_mismatch",   # meter reading vs billed units inconsistency
    "spike_anomaly",      # unexpected spike (illegal usage)
    "zero_consumption",   # zero / near-zero billing despite active connection
    "repeated_low",       # consistently low across several cycles
]


def pick_consumer_type():
    types = list(CONSUMER_TYPES.keys())
    weights = [CONSUMER_TYPES[t]["weight"] for t in types]
    return random.choices(types, weights=weights, k=1)[0]


def generate_normal_consumption(ctype: str, months: int) -> list[float]:
    cfg = CONSUMER_TYPES[ctype]
    base = cfg["base"]
    std = cfg["std"]
    # Add slight seasonal variation (higher in summer months 4-8)
    seasonals = []
    for m in range(months):
        month_idx = m % 12
        seasonal = 1.0 + 0.15 * np.sin(np.pi * month_idx / 6)
        val = max(10, np.random.normal(base * seasonal, std))
        seasonals.append(round(val, 1))
    return seasonals


def inject_fraud(consumption: list[float], fraud_type: str) -> tuple[list[float], list[str]]:
    """Return modified consumption list and list of affected month indices."""
    n = len(consumption)
    affected = []

    if fraud_type == "sudden_drop":
        # Months 6-9: consumption drops by 75-90%
        start = random.randint(4, 6)
        for i in range(start, min(start + 4, n)):
            consumption[i] = round(consumption[i] * random.uniform(0.05, 0.20), 1)
            affected.append(i)

    elif fraud_type == "spike_anomaly":
        # Two random months have 3-5× spikes
        idxs = random.sample(range(2, n), min(2, n - 2))
        for i in idxs:
            consumption[i] = round(consumption[i] * random.uniform(3.0, 5.5), 1)
            affected.append(i)

    elif fraud_type == "zero_consumption":
        # 3-5 consecutive near-zero months
        start = random.randint(3, 6)
        for i in range(start, min(start + random.randint(3, 5), n)):
            consumption[i] = round(random.uniform(1, 8), 1)
            affected.append(i)

    elif fraud_type == "repeated_low":
        # Every other month is ~85% lower than normal
        for i in range(1, n, 2):
            consumption[i] = round(consumption[i] * random.uniform(0.10, 0.20), 1)
            affected.append(i)

    # reading_mismatch is handled later at the bill-row level
    return consumption, affected


def build_dataset():
    records = []
    consumer_meta = []

    fraud_consumer_count = int(NUM_CONSUMERS * FRAUD_FRACTION)
    fraud_pool = list(range(fraud_consumer_count))
    random.shuffle(fraud_pool)
    fraud_assignments = {}
    for cid_idx in fraud_pool:
        fraud_assignments[cid_idx] = random.choice(FRAUD_TYPES)

    start_date = datetime(2023, 1, 1)

    for c_idx in range(NUM_CONSUMERS):
        consumer_id = f"C{100 + c_idx:03d}"
        meter_id = f"M{2000 + c_idx:04d}"
        ctype = pick_consumer_type()
        city_zone = random.choice(["Zone-A", "Zone-B", "Zone-C", "Zone-D"])
        tariff_category = ctype.split("_")[0].capitalize()

        consumption = generate_normal_consumption(ctype, MONTHS)
        fraud_type = fraud_assignments.get(c_idx)

        affected_months = []
        if fraud_type and fraud_type != "reading_mismatch":
            consumption, affected_months = inject_fraud(consumption[:], fraud_type)
        elif fraud_type == "reading_mismatch":
            affected_months = random.sample(range(2, MONTHS), min(3, MONTHS - 2))

        # Build meter readings cumulatively
        initial_reading = random.randint(1000, 50000)
        meter_readings = [initial_reading]
        for units in consumption:
            meter_readings.append(meter_readings[-1] + units)

        consumer_meta.append({
            "consumer_id": consumer_id,
            "meter_id": meter_id,
            "consumer_type": ctype,
            "tariff_category": tariff_category,
            "city_zone": city_zone,
            "fraud_type": fraud_type or "none",
        })

        for m in range(MONTHS):
            billing_date = start_date + timedelta(days=30 * m)
            billing_month = billing_date.strftime("%Y-%m")
            prev_reading = round(meter_readings[m], 1)
            curr_reading = round(meter_readings[m + 1], 1)
            actual_units = round(curr_reading - prev_reading, 1)

            # Inject reading-mismatch fraud
            billed_units = actual_units
            mismatch_flag = False
            if fraud_type == "reading_mismatch" and m in affected_months:
                # Underreport billed units vs actual meter diff
                billed_units = round(actual_units * random.uniform(0.25, 0.50), 1)
                mismatch_flag = True

            bill_amount = round(billed_units * RATE_PER_UNIT, 2)

            # Occasionally add small data-quality issues (NaN, duplicates)
            if random.random() < 0.005:
                bill_amount = None   # missing amount
            if random.random() < 0.003:
                billed_units = None  # missing billed units

            records.append({
                "consumer_id": consumer_id,
                "meter_id": meter_id,
                "billing_month": billing_month,
                "billing_date": billing_date.strftime("%Y-%m-%d"),
                "previous_reading": prev_reading,
                "current_reading": curr_reading,
                "actual_units_from_reading": actual_units,
                "billed_units": billed_units,
                "bill_amount": bill_amount,
                "consumer_type": ctype,
                "tariff_category": tariff_category,
                "city_zone": city_zone,
                "is_fraud_injected": (m in affected_months) and (fraud_type is not None),
                "injected_fraud_type": fraud_type or "none",
            })

    df = pd.DataFrame(records)
    meta_df = pd.DataFrame(consumer_meta)
    return df, meta_df


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    df, meta_df = build_dataset()
    df.to_csv(os.path.join(out_dir, "electricity_bills.csv"), index=False)
    meta_df.to_csv(os.path.join(out_dir, "consumer_metadata.csv"), index=False)

    print(f"Generated {len(df)} billing records for {len(meta_df)} consumers.")
    print(f"Fraud consumers: {(meta_df['fraud_type'] != 'none').sum()}")
    print(f"Files saved to: {out_dir}")
