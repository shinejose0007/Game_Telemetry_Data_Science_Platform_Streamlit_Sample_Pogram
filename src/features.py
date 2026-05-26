import pandas as pd
import numpy as np


def daily_device_features(sessions: pd.DataFrame, maintenance: pd.DataFrame | None = None) -> pd.DataFrame:
    if sessions.empty:
        return pd.DataFrame()

    df = sessions.copy()
    df["date"] = pd.to_datetime(df["session_start"]).dt.date
    df["session_duration_min"] = (
        pd.to_datetime(df["session_end"]) - pd.to_datetime(df["session_start"])
    ).dt.total_seconds() / 60

    agg = (
        df.groupby(
            [
                "date",
                "device_id",
                "city",
                "region",
                "location_type",
                "device_type",
                "software_version",
            ],
            as_index=False,
        )
        .agg(
            sessions=("session_id", "count"),
            total_revenue=("revenue_eur", "sum"),
            avg_revenue=("revenue_eur", "mean"),
            total_plays=("plays", "sum"),
            total_errors=("error_count", "sum"),
            avg_duration_min=("session_duration_min", "mean"),
            unique_games=("game_title", "nunique"),
        )
    )

    agg["error_rate"] = agg["total_errors"] / agg["total_plays"].replace(0, np.nan)
    agg["revenue_per_session"] = agg["total_revenue"] / agg["sessions"].replace(0, np.nan)
    agg["date"] = pd.to_datetime(agg["date"])
    agg["day_of_week"] = agg["date"].dt.weekday
    agg["is_weekend"] = (agg["day_of_week"] >= 5).astype(int)
    agg["month"] = agg["date"].dt.month
    agg["week"] = agg["date"].dt.isocalendar().week.astype(int)

    # target for next-day forecasting
    agg = agg.sort_values(["device_id", "date"])
    agg["next_day_revenue"] = agg.groupby("device_id")["total_revenue"].shift(-1)

    # maintenance target within 14 days after current date
    agg["maintenance_next_14d"] = 0
    if maintenance is not None and not maintenance.empty:
        maint = maintenance.copy()
        maint["maintenance_date"] = pd.to_datetime(maint["maintenance_date"])
        maint_by_device = maint.groupby("device_id")["maintenance_date"].apply(list).to_dict()
        targets = []
        for _, row in agg.iterrows():
            future_dates = maint_by_device.get(row["device_id"], [])
            target = any(0 <= (mdate - row["date"]).days <= 14 for mdate in future_dates)
            targets.append(int(target))
        agg["maintenance_next_14d"] = targets

    return agg


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["check", "status", "details"])

    checks = []

    def add(check, status, details):
        checks.append({"check": check, "status": status, "details": details})

    missing = df.isna().sum()
    missing_total = int(missing.sum())
    add("Missing values", "PASS" if missing_total == 0 else "WARN", f"{missing_total} missing cells")

    duplicate_sessions = int(df["session_id"].duplicated().sum()) if "session_id" in df else 0
    add("Duplicate session IDs", "PASS" if duplicate_sessions == 0 else "FAIL", f"{duplicate_sessions} duplicates")

    negative_revenue = int((df["revenue_eur"] < 0).sum()) if "revenue_eur" in df else 0
    add("Negative revenue", "PASS" if negative_revenue == 0 else "FAIL", f"{negative_revenue} rows")

    negative_plays = int((df["plays"] < 0).sum()) if "plays" in df else 0
    add("Negative plays", "PASS" if negative_plays == 0 else "FAIL", f"{negative_plays} rows")

    very_high_errors = int((df["error_count"] > 15).sum()) if "error_count" in df else 0
    add("Very high error count", "PASS" if very_high_errors == 0 else "WARN", f"{very_high_errors} rows with error_count > 15")

    if "session_start" in df and "session_end" in df:
        durations = (pd.to_datetime(df["session_end"]) - pd.to_datetime(df["session_start"])).dt.total_seconds() / 60
        invalid_durations = int((durations <= 0).sum())
        add("Invalid session duration", "PASS" if invalid_durations == 0 else "FAIL", f"{invalid_durations} rows")

    return pd.DataFrame(checks)


def executive_kpis(sessions: pd.DataFrame) -> dict:
    if sessions.empty:
        return {
            "total_revenue": 0,
            "total_sessions": 0,
            "active_devices": 0,
            "avg_revenue_per_session": 0,
            "total_errors": 0,
            "error_rate": 0,
        }
    total_revenue = sessions["revenue_eur"].sum()
    total_sessions = len(sessions)
    active_devices = sessions["device_id"].nunique()
    total_errors = sessions["error_count"].sum()
    total_plays = sessions["plays"].sum()
    return {
        "total_revenue": total_revenue,
        "total_sessions": total_sessions,
        "active_devices": active_devices,
        "avg_revenue_per_session": total_revenue / total_sessions if total_sessions else 0,
        "total_errors": total_errors,
        "error_rate": total_errors / total_plays if total_plays else 0,
    }
