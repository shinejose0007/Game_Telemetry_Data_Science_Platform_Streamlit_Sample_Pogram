from datetime import datetime
import pandas as pd


def simulate_azure_pipeline_status() -> pd.DataFrame:
    """Simulates Azure Data Factory style pipeline monitoring."""
    return pd.DataFrame(
        [
            {
                "pipeline_name": "ingest_game_sessions",
                "source": "Device telemetry stream",
                "target": "SQLite raw layer",
                "status": "Succeeded",
                "last_run": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "records_processed": 12540,
            },
            {
                "pipeline_name": "build_daily_device_kpis",
                "source": "Raw session table",
                "target": "Feature layer",
                "status": "Succeeded",
                "last_run": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "records_processed": 21480,
            },
            {
                "pipeline_name": "train_ml_models",
                "source": "Feature layer",
                "target": "Model registry",
                "status": "Ready",
                "last_run": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "records_processed": 0,
            },
        ]
    )


def simulate_sap_datasphere_model() -> pd.DataFrame:
    """Simulates a SAP Datasphere-style semantic model."""
    return pd.DataFrame(
        [
            {"business_object": "Device", "key_field": "device_id", "measure_or_dimension": "Dimension"},
            {"business_object": "Location", "key_field": "location_id", "measure_or_dimension": "Dimension"},
            {"business_object": "Game Session", "key_field": "session_id", "measure_or_dimension": "Fact"},
            {"business_object": "Revenue", "key_field": "revenue_eur", "measure_or_dimension": "Measure"},
            {"business_object": "Error Events", "key_field": "error_count", "measure_or_dimension": "Measure"},
            {"business_object": "Maintenance", "key_field": "maintenance_id", "measure_or_dimension": "Fact"},
        ]
    )
