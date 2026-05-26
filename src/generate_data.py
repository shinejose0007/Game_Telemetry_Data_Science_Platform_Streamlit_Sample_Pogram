from datetime import datetime, timedelta
import random
import numpy as np
import pandas as pd

from .database import init_db, write_df, table_count

RNG = np.random.default_rng(42)
random.seed(42)


def create_synthetic_data(
    n_devices: int = 120,
    n_days: int = 180,
    avg_sessions_per_device_day: float = 4.5
):
    cities = [
        ("Lübbecke", "NRW", "Entertainment Center"),
        ("Bielefeld", "NRW", "Arcade Partner"),
        ("Hannover", "Lower Saxony", "Entertainment Center"),
        ("Dortmund", "NRW", "Arcade Partner"),
        ("Münster", "NRW", "Retail Partner"),
        ("Düsseldorf", "NRW", "Entertainment Center"),
        ("Hamburg", "Hamburg", "Retail Partner"),
        ("Bremen", "Bremen", "Arcade Partner"),
    ]

    locations = []
    for i, (city, region, loc_type) in enumerate(cities, start=1):
        locations.append(
            {
                "location_id": f"L{i:03d}",
                "city": city,
                "region": region,
                "location_type": loc_type,
            }
        )
    locations_df = pd.DataFrame(locations)

    software_versions_df = pd.DataFrame(
        [
            {
                "software_version": "v1.8",
                "release_date": (datetime.today() - timedelta(days=400)).date().isoformat(),
                "version_notes": "Legacy stable release",
            },
            {
                "software_version": "v2.0",
                "release_date": (datetime.today() - timedelta(days=250)).date().isoformat(),
                "version_notes": "Performance improvements",
            },
            {
                "software_version": "v2.1",
                "release_date": (datetime.today() - timedelta(days=120)).date().isoformat(),
                "version_notes": "New telemetry module",
            },
            {
                "software_version": "v2.2",
                "release_date": (datetime.today() - timedelta(days=45)).date().isoformat(),
                "version_notes": "R&D test release with new balancing logic",
            },
        ]
    )

    device_types = ["Cabinet Classic", "Cabinet Pro", "Touch Terminal", "Compact Terminal"]
    statuses = ["active", "active", "active", "active", "maintenance", "inactive"]
    software_versions = ["v1.8", "v2.0", "v2.1", "v2.2"]

    devices = []
    for i in range(1, n_devices + 1):
        install_date = datetime.today() - timedelta(days=int(RNG.integers(90, 1200)))
        devices.append(
            {
                "device_id": f"D{i:04d}",
                "location_id": random.choice(locations_df["location_id"].tolist()),
                "device_type": random.choice(device_types),
                "install_date": install_date.date().isoformat(),
                "software_version": random.choices(software_versions, weights=[0.15, 0.25, 0.35, 0.25])[0],
                "status": random.choices(statuses, weights=[0.82, 0.03, 0.03, 0.03, 0.06, 0.03])[0],
            }
        )
    devices_df = pd.DataFrame(devices)

    game_catalog = [
        ("Treasure Quest", "Adventure", 1.25),
        ("Fruit Galaxy", "Classic", 1.05),
        ("Mystic Wheels", "Chance", 1.35),
        ("Neon Racer", "Skill", 1.10),
        ("Lucky Harbor", "Classic", 1.20),
        ("Crystal Maze", "Puzzle", 0.95),
        ("Space Spin", "Arcade", 1.15),
        ("Golden Safari", "Adventure", 1.30),
    ]
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]

    start_date = datetime.today().date() - timedelta(days=n_days)
    sessions = []
    sid = 1

    for day in range(n_days):
        current_date = start_date + timedelta(days=day)
        weekday = current_date.weekday()
        weekend_multiplier = 1.45 if weekday >= 5 else 1.0

        for _, device in devices_df.iterrows():
            if device["status"] == "inactive":
                continue
            base_lambda = avg_sessions_per_device_day * weekend_multiplier
            if device["device_type"] == "Cabinet Pro":
                base_lambda *= 1.2
            if device["software_version"] == "v2.2":
                base_lambda *= 1.08

            n_sessions = int(RNG.poisson(base_lambda))
            for _ in range(n_sessions):
                game_title, category, price_factor = random.choice(game_catalog)
                hour = int(np.clip(RNG.normal(18 if weekday >= 5 else 16, 4), 8, 23))
                minute = int(RNG.integers(0, 60))
                session_start = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=hour, minutes=minute)
                duration = int(np.clip(RNG.normal(18, 8), 3, 90))
                session_end = session_start + timedelta(minutes=duration)
                plays = max(1, int(RNG.poisson(duration / 6) + 1))

                location_factor = 1.0
                city = locations_df.loc[locations_df["location_id"] == device["location_id"], "city"].iloc[0]
                if city in ["Düsseldorf", "Hamburg", "Hannover"]:
                    location_factor = 1.18

                sw_factor = {"v1.8": 0.95, "v2.0": 1.00, "v2.1": 1.05, "v2.2": 1.10}[device["software_version"]]
                revenue = plays * price_factor * location_factor * sw_factor * float(RNG.normal(1.0, 0.12))
                revenue = max(0.5, revenue)

                base_error_rate = {"v1.8": 0.035, "v2.0": 0.025, "v2.1": 0.030, "v2.2": 0.045}[device["software_version"]]
                if device["status"] == "maintenance":
                    base_error_rate += 0.08
                error_count = int(RNG.poisson(base_error_rate * plays * 3))

                # Create occasional anomalies: sudden revenue drop or error spike.
                if RNG.random() < 0.006:
                    revenue *= float(RNG.uniform(0.05, 0.30))
                    error_count += int(RNG.integers(3, 9))
                    event_log = "warning: abnormal revenue drop and elevated error count"
                elif error_count >= 3:
                    event_log = random.choice(
                        [
                            "warning: repeated user-interface timeout",
                            "warning: coin validator retry event",
                            "warning: network synchronization delay",
                            "info: session completed with recoverable device event",
                        ]
                    )
                else:
                    event_log = random.choice(
                        [
                            "info: session completed normally",
                            "info: standard telemetry event",
                            "info: no device exception",
                        ]
                    )

                sessions.append(
                    {
                        "session_id": f"S{sid:08d}",
                        "device_id": device["device_id"],
                        "session_start": session_start.isoformat(timespec="seconds"),
                        "session_end": session_end.isoformat(timespec="seconds"),
                        "game_title": game_title,
                        "game_category": category,
                        "plays": plays,
                        "revenue_eur": round(float(revenue), 2),
                        "error_count": error_count,
                        "age_group": random.choices(age_groups, weights=[0.18, 0.32, 0.25, 0.17, 0.08])[0],
                        "event_log": event_log,
                    }
                )
                sid += 1

    sessions_df = pd.DataFrame(sessions)

    maintenance = []
    mid = 1
    for _, device in devices_df.iterrows():
        device_sessions = sessions_df[sessions_df["device_id"] == device["device_id"]]
        total_errors = int(device_sessions["error_count"].sum()) if not device_sessions.empty else 0
        maintenance_probability = min(0.85, 0.08 + total_errors / 550)
        if RNG.random() < maintenance_probability:
            n_events = 1 + int(RNG.random() < 0.2)
            for _ in range(n_events):
                maintenance_date = start_date + timedelta(days=int(RNG.integers(5, n_days)))
                maintenance.append(
                    {
                        "maintenance_id": f"M{mid:06d}",
                        "device_id": device["device_id"],
                        "maintenance_date": maintenance_date.isoformat(),
                        "maintenance_type": random.choice(["preventive", "corrective", "software update", "hardware inspection"]),
                        "downtime_minutes": int(RNG.integers(20, 240)),
                        "technician_note": random.choice(
                            [
                                "Checked error logs and reset device components.",
                                "Software patch installed after telemetry warning.",
                                "Preventive inspection completed.",
                                "High error count required detailed inspection.",
                            ]
                        ),
                    }
                )
                mid += 1

    maintenance_df = pd.DataFrame(maintenance)

    return {
        "locations": locations_df,
        "software_versions": software_versions_df,
        "devices": devices_df,
        "game_sessions": sessions_df,
        "maintenance_logs": maintenance_df,
    }


def ensure_demo_data(force: bool = False) -> None:
    init_db()
    if not force and table_count("game_sessions") > 0:
        return

    data = create_synthetic_data()
    for table, df in data.items():
        df.to_csv(f"data/{table}.csv", index=False)
        write_df(df, table, if_exists="replace")
