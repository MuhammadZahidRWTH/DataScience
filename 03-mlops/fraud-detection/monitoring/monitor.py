import os
import json
import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

REFERENCE_PATH = "data/transactions.csv"
CURRENT_PATH   = "data/transactions_drift.csv"
REPORT_DIR     = "monitoring/reports"

# these are the features i care most about drifting
# transaction_amount and velocity are the most predictive in the model
# (check model/feature_importance.json) so if those drift, model performance
# will likely drop before we even notice in metrics
CRITICAL_FEATURES = [
    "transaction_amount",
    "velocity_24h",
    "distance_from_home",
    "utilization_rate",
    "failed_attempts_24h",
]


def load_data():
    if not os.path.exists(REFERENCE_PATH):
        raise FileNotFoundError(
            f"reference data not found at {REFERENCE_PATH} — run data/generate_data.py first"
        )
    if not os.path.exists(CURRENT_PATH):
        raise FileNotFoundError(
            f"current data not found at {CURRENT_PATH} — run data/generate_data.py first"
        )

    reference = pd.read_csv(REFERENCE_PATH)
    current   = pd.read_csv(CURRENT_PATH)

    if len(current) < 500:
        print(f"warning: current dataset only has {len(current)} rows — drift results may be unreliable")

    return reference, current


def run_drift_report():
    os.makedirs(REPORT_DIR, exist_ok=True)

    reference, current = load_data()
    print(f"reference: {len(reference)} rows | current: {len(current)} rows")

    # define data schema
    definition = DataDefinition(
        numerical_columns=CRITICAL_FEATURES
     
    )

    ref_dataset = Dataset.from_pandas(reference, data_definition=definition)
    cur_dataset = Dataset.from_pandas(current,   data_definition=definition)

    # run drift report
    report = Report([DataDriftPreset()])
    run    = report.run(reference_data=ref_dataset, current_data=cur_dataset)

    # save html
    run.save_html(f"{REPORT_DIR}/drift_report.html")
    print(f"html report saved → {REPORT_DIR}/drift_report.html")

    # save json summary
    result = run.dict()
    with open(f"{REPORT_DIR}/drift_summary.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print("\n✅ drift report complete")
    print("   open monitoring/reports/drift_report.html for the full breakdown")

    return result


if __name__ == "__main__":
    run_drift_report()