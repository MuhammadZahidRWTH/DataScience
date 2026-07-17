# data/raw/generate_data.py
# Generates synthetic industrial sensor data
# Mimics real manufacturing sensor readings (vibration, temperature, pressure)
# Based on SECOM-style datasets used in Industry 4.0 research

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

np.random.seed(42)

def generate_sensor_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic sensor readings for 3 industrial machines.
    Includes normal operation + injected fault patterns.
    """
    timestamps = [
        datetime(2024, 1, 1) + timedelta(hours=i)
        for i in range(n_samples)
    ]

    machines = ["CNC_Mill_01", "Hydraulic_Press_02", "Conveyor_Motor_03"]
    records = []

    for i, ts in enumerate(timestamps):
        for machine in machines:
            # Normal baseline varies per machine
            base_temp = {"CNC_Mill_01": 65, "Hydraulic_Press_02": 78, "Conveyor_Motor_03": 45}
            base_vib = {"CNC_Mill_01": 0.8, "Hydraulic_Press_02": 1.2, "Conveyor_Motor_03": 0.5}
            base_pressure = {"CNC_Mill_01": 4.2, "Hydraulic_Press_02": 8.5, "Conveyor_Motor_03": 2.1}

            # inject fault patterns every ~500 hours per machine
            fault_flag = 0
            fault_type = "none"

            if i % 500 == 0 and i > 0:
                # bearing wear — vibration spike
                vibration = base_vib[machine] + np.random.normal(3.5, 0.4)
                temperature = base_temp[machine] + np.random.normal(12, 2)
                pressure = base_pressure[machine] + np.random.normal(0.3, 0.1)
                fault_flag = 1
                fault_type = "bearing_wear"
            elif i % 300 == 0 and i > 0:
                # overheating
                vibration = base_vib[machine] + np.random.normal(0.2, 0.1)
                temperature = base_temp[machine] + np.random.normal(22, 3)
                pressure = base_pressure[machine] + np.random.normal(0.1, 0.05)
                fault_flag = 1
                fault_type = "overheating"
            else:
                # normal operation with realistic noise
                vibration = base_vib[machine] + np.random.normal(0, 0.08)
                temperature = base_temp[machine] + np.random.normal(0, 1.5)
                pressure = base_pressure[machine] + np.random.normal(0, 0.15)

            records.append({
                "timestamp": ts,
                "machine_id": machine,
                "temperature_c": round(temperature, 2),
                "vibration_mm_s": round(vibration, 3),
                "pressure_bar": round(pressure, 3),
                "rpm": round(np.random.normal(1450, 20), 0),
                "fault_flag": fault_flag,
                "fault_type": fault_type,
            })

    df = pd.DataFrame(records)
    return df


def generate_maintenance_logs(n_logs: int = 200) -> pd.DataFrame:
    """
    Generate synthetic maintenance log entries.
    These are the messy text logs LLM agents will clean and interpret.
    """
    fault_descriptions = [
        "Machine vibration exceeded threshold. Bearing replacement required.",
        "Temp sensor reading 92C. Cooling fan inspection needed.",
        "Hydraulic pressure drop detected. Seal inspection scheduled.",
        "Unusual noise from motor. Lubrication applied. Monitor closely.",
        "Scheduled maintenance completed. All systems nominal.",
        "Emergency stop triggered. Operator reported burning smell.",
        "Vibration spike on spindle. Realignment performed.",
        "Oil level low. Topped up. Next service in 200h.",
        "Filter clogged. Replaced. Pressure restored to normal.",
        "Gearbox temperature high. Coolant flow checked and adjusted.",
    ]

    machines = ["CNC_Mill_01", "Hydraulic_Press_02", "Conveyor_Motor_03"]
    technicians = ["Hans Mueller", "Anna Schmidt", "Karl Weber", "Lisa Fischer"]

    logs = []
    base_date = datetime(2024, 1, 1)

    for i in range(n_logs):
        logs.append({
            "log_id": f"LOG_{i+1:04d}",
            "timestamp": base_date + timedelta(days=i * 1.8),
            "machine_id": np.random.choice(machines),
            "technician": np.random.choice(technicians),
            "description": np.random.choice(fault_descriptions),
            "downtime_hours": round(np.random.exponential(2.5), 1),
            "resolved": np.random.choice([True, False], p=[0.85, 0.15]),
        })

    return pd.DataFrame(logs)


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating sensor data...")
    sensor_df = generate_sensor_data()
    sensor_df.to_csv(output_dir / "sensor_readings.csv", index=False)
    print(f"  sensor_readings.csv — {len(sensor_df)} rows")

    print("Generating maintenance logs...")
    logs_df = generate_maintenance_logs()
    logs_df.to_csv(output_dir / "maintenance_logs.csv", index=False)
    print(f"  maintenance_logs.csv — {len(logs_df)} rows")

    print("Done. Data ready for MaintenanceGPT agents.")
