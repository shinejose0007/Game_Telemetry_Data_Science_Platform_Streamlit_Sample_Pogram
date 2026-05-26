from pathlib import Path
import sqlite3
import pandas as pd

DB_PATH = Path("data/game_telemetry.db")


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_db(db_path: Path = DB_PATH) -> None:
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS locations (
            location_id TEXT PRIMARY KEY,
            city TEXT NOT NULL,
            region TEXT NOT NULL,
            location_type TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS software_versions (
            software_version TEXT PRIMARY KEY,
            release_date TEXT NOT NULL,
            version_notes TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS devices (
            device_id TEXT PRIMARY KEY,
            location_id TEXT NOT NULL,
            device_type TEXT NOT NULL,
            install_date TEXT NOT NULL,
            software_version TEXT NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY(location_id) REFERENCES locations(location_id),
            FOREIGN KEY(software_version) REFERENCES software_versions(software_version)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS game_sessions (
            session_id TEXT PRIMARY KEY,
            device_id TEXT NOT NULL,
            session_start TEXT NOT NULL,
            session_end TEXT NOT NULL,
            game_title TEXT NOT NULL,
            game_category TEXT NOT NULL,
            plays INTEGER NOT NULL,
            revenue_eur REAL NOT NULL,
            error_count INTEGER NOT NULL,
            age_group TEXT NOT NULL,
            event_log TEXT,
            FOREIGN KEY(device_id) REFERENCES devices(device_id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS maintenance_logs (
            maintenance_id TEXT PRIMARY KEY,
            device_id TEXT NOT NULL,
            maintenance_date TEXT NOT NULL,
            maintenance_type TEXT NOT NULL,
            downtime_minutes INTEGER NOT NULL,
            technician_note TEXT,
            FOREIGN KEY(device_id) REFERENCES devices(device_id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS model_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_time TEXT NOT NULL,
            model_name TEXT NOT NULL,
            target TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            notes TEXT
        )
        """
    )

    conn.commit()
    conn.close()


def table_count(table: str, db_path: Path = DB_PATH) -> int:
    conn = get_connection(db_path)
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
    except sqlite3.OperationalError:
        count = 0
    conn.close()
    return count


def write_df(df: pd.DataFrame, table: str, if_exists: str = "replace", db_path: Path = DB_PATH) -> None:
    conn = get_connection(db_path)
    df.to_sql(table, conn, if_exists=if_exists, index=False)
    conn.close()


def read_sql(query: str, params=None, db_path: Path = DB_PATH) -> pd.DataFrame:
    conn = get_connection(db_path)
    df = pd.read_sql_query(query, conn, params=params or {})
    conn.close()
    return df


def read_table(table: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    return read_sql(f"SELECT * FROM {table}", db_path=db_path)


def load_joined_sessions(db_path: Path = DB_PATH) -> pd.DataFrame:
    query = """
    SELECT
        s.session_id,
        s.device_id,
        s.session_start,
        s.session_end,
        s.game_title,
        s.game_category,
        s.plays,
        s.revenue_eur,
        s.error_count,
        s.age_group,
        s.event_log,
        d.location_id,
        d.device_type,
        d.install_date,
        d.software_version,
        d.status,
        l.city,
        l.region,
        l.location_type
    FROM game_sessions s
    JOIN devices d ON s.device_id = d.device_id
    JOIN locations l ON d.location_id = l.location_id
    """
    df = read_sql(query, db_path=db_path)
    if not df.empty:
        df["session_start"] = pd.to_datetime(df["session_start"])
        df["session_end"] = pd.to_datetime(df["session_end"])
        df["date"] = df["session_start"].dt.date
        df["hour"] = df["session_start"].dt.hour
        df["weekday"] = df["session_start"].dt.day_name()
        df["month"] = df["session_start"].dt.to_period("M").astype(str)
    return df
