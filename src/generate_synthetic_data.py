"""
generate_synthetic_data.py

Generates a synthetic private-credit style loan portfolio dataset:
borrower -> facility -> transaction (payment history) grain.

This mimics the kind of multi-level portfolio data (borrower, facility,
collateral, transaction) used in private credit / asset-based lending
reporting, scaled down to something a single-machine PySpark pipeline
can process end to end.

Run:
    python src/generate_synthetic_data.py
"""

import random
import csv
import os
from datetime import date, timedelta

random.seed(42)

N_BORROWERS = 2000
N_FACILITIES_PER_BORROWER = (1, 2)  # min, max
MONTHS_OF_HISTORY = 24

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(OUT_DIR, exist_ok=True)

INDUSTRIES = [
    "Manufacturing", "Retail", "Healthcare", "Technology",
    "Real Estate", "Logistics", "Hospitality", "Energy",
]

RISK_TIERS = ["Prime", "Near-Prime", "Subprime"]

# Base default probability by risk tier — drives simulated payment behavior
TIER_DEFAULT_RATE = {"Prime": 0.02, "Near-Prime": 0.08, "Subprime": 0.20}


def random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def generate_borrowers(n):
    rows = []
    for i in range(1, n + 1):
        tier = random.choices(RISK_TIERS, weights=[0.5, 0.35, 0.15])[0]
        rows.append({
            "borrower_id": f"B{i:06d}",
            "industry": random.choice(INDUSTRIES),
            "risk_tier": tier,
            "annual_revenue": round(random.uniform(0.5, 500) * 1_000_000, 2),
            "years_in_business": random.randint(1, 40),
            "onboarding_date": random_date(date(2019, 1, 1), date(2024, 1, 1)).isoformat(),
        })
    return rows


def generate_facilities(borrowers):
    rows = []
    fac_counter = 1
    for b in borrowers:
        n_fac = random.randint(*N_FACILITIES_PER_BORROWER)
        for _ in range(n_fac):
            limit = round(random.uniform(0.2, 20) * 1_000_000, 2)
            disb_date = random_date(date(2022, 1, 1), date(2024, 6, 1))
            rows.append({
                "facility_id": f"F{fac_counter:07d}",
                "borrower_id": b["borrower_id"],
                "facility_type": random.choice(["Term Loan", "Revolver", "ABL"]),
                "credit_limit": limit,
                "interest_rate": round(random.uniform(0.06, 0.16), 4),
                "disbursement_date": disb_date.isoformat(),
                "maturity_date": (disb_date + timedelta(days=365 * random.randint(2, 7))).isoformat(),
                "collateral_value": round(limit * random.uniform(1.0, 1.8), 2),
            })
            fac_counter += 1
    return rows


def generate_transactions(facilities, borrower_tier_map):
    """Monthly payment/exposure snapshots per facility for MONTHS_OF_HISTORY months."""
    rows = []
    for f in facilities:
        tier = borrower_tier_map[f["borrower_id"]]
        default_rate = TIER_DEFAULT_RATE[tier]
        outstanding = f["credit_limit"] * random.uniform(0.4, 0.95)
        dpd_state = 0  # days past due, evolves month to month
        disb = date.fromisoformat(f["disbursement_date"])

        for m in range(MONTHS_OF_HISTORY):
            period = disb + timedelta(days=30 * m)
            if period > date(2024, 12, 31):
                break

            scheduled_payment = round(f["credit_limit"] * 0.03, 2)

            # simulate missed payment probability based on risk tier + current dpd
            miss_prob = default_rate + (0.15 if dpd_state > 0 else 0)
            missed = random.random() < miss_prob

            if missed:
                dpd_state += 30
                actual_payment = 0.0
            else:
                # partial catch-up if previously behind
                dpd_state = max(0, dpd_state - 30) if dpd_state > 0 else 0
                actual_payment = scheduled_payment

            outstanding = max(0.0, outstanding - actual_payment + random.uniform(-5000, 15000))

            rows.append({
                "facility_id": f["facility_id"],
                "period_end_date": period.isoformat(),
                "scheduled_payment": scheduled_payment,
                "actual_payment": round(actual_payment, 2),
                "outstanding_exposure": round(outstanding, 2),
                "days_past_due": dpd_state,
            })
    return rows


def write_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):,} rows -> {path}")


def main():
    borrowers = generate_borrowers(N_BORROWERS)
    facilities = generate_facilities(borrowers)
    tier_map = {b["borrower_id"]: b["risk_tier"] for b in borrowers}
    transactions = generate_transactions(facilities, tier_map)

    write_csv(borrowers, os.path.join(OUT_DIR, "borrowers.csv"), list(borrowers[0].keys()))
    write_csv(facilities, os.path.join(OUT_DIR, "facilities.csv"), list(facilities[0].keys()))
    write_csv(transactions, os.path.join(OUT_DIR, "transactions.csv"), list(transactions[0].keys()))


if __name__ == "__main__":
    main()
