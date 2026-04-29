"""Generate a sample dataset for the end-to-end ML pipeline project."""

from __future__ import annotations

import csv
from pathlib import Path
from random import Random


OUTPUT_PATH = Path("data") / "training_data.csv"


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rng = Random(42)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["age", "income", "tenure_months", "product", "region", "target"])
        for _ in range(2500):
            age = rng.randint(21, 70)
            income = rng.randint(25000, 180000)
            tenure_months = rng.randint(1, 120)
            product = rng.choice(["standard", "premium", "gold"])
            region = rng.choice(["east", "west", "north", "south"])

            score = 0.0
            score += 1.0 if income > 90000 else 0.0
            score += 0.8 if tenure_months > 24 else 0.0
            score += 0.7 if product in {"premium", "gold"} else 0.0
            score += 0.4 if age >= 35 else 0.0
            target = 1 if rng.random() < min(score / 3.5, 0.95) else 0

            writer.writerow([age, income, tenure_months, product, region, target])
    print(f"Saved sample dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
