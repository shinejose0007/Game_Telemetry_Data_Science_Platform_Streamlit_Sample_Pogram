import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from src.auth import init_auth_db, authenticate, register_user, list_users
from src.database import init_db, load_joined_sessions, read_table, read_sql
from src.generate_data import ensure_demo_data
from src.features import daily_device_features, data_quality_report, executive_kpis
from src.models import train_revenue_forecast, train_maintenance_classifier, train_anomaly_detector
from src.reports import create_excel_report, create_pdf_summary
from src.azure_sap_simulation import simulate_azure_pipeline_status, simulate_sap_datasphere_model

st.set_page_config(
    page_title="Game Telemetry Data Science Platform",
    page_icon="🎮",
    layout="wide",
)

DB_PATH = "data/game_telemetry.db"


# -----------------------------
# Initialization
# -----------------------------
Path("data").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)
init_db()
init_auth_db()
ensure_demo_data(force=False)


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data():
    sessions = load_joined_sessions()
    maintenance = read_table("maintenance_logs")
    devices = read_table("devices")
    model_runs = read_sql("SELECT * FROM model_runs ORDER BY run_time DESC")
    features = daily_device_features(sessions, maintenance)
    return sessions, maintenance, devices, model_runs, features


def reset_cache():
    st.cache_data.clear()


def metric_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)


def download_file_button(path: Path, label: str, mime: str):
    with open(path, "rb") as f:
        st.download_button(label=label, data=f, file_name=path.name, mime=mime)


def restrict_page(allowed_roles):
    user = st.session_state.get("user")
    if not user or user["role"] not in allowed_roles:
        st.warning("You do not have access to this page with your current role.")
        st.stop()


def apply_filters(df: pd.DataFrame):
    if df.empty:
        return df

    with st.sidebar.expander("Data filters", expanded=True):
        min_date = pd.to_datetime(df["session_start"]).min().date()
        max_date = pd.to_datetime(df["session_start"]).max().date()
        date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

        cities = sorted(df["city"].dropna().unique().tolist())
        selected_cities = st.multiselect("City", cities, default=cities)

        games = sorted(df["game_title"].dropna().unique().tolist())
        selected_games = st.multiselect("Game title", games, default=games)

        versions = sorted(df["software_version"].dropna().unique().tolist())
        selected_versions = st.multiselect("Software version", versions, default=versions)

    filtered = df.copy()
    if len(date_range) == 2:
        start, end = date_range
        filtered = filtered[
            (filtered["session_start"].dt.date >= start) &
            (filtered["session_start"].dt.date <= end)
        ]
    filtered = filtered[
        filtered["city"].isin(selected_cities) &
        filtered["game_title"].isin(selected_games) &
        filtered["software_version"].isin(selected_versions)
    ]
    return filtered


# -----------------------------
# Auth UI
# -----------------------------
def login_screen():
    st.title("🎮 Game Telemetry Data Science Platform")
    st.caption("Portfolio project: Python, SQL, Machine Learning, Statistics, Streamlit, Reporting")

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Login")
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", type="password", value="admin123")
        if st.button("Login", type="primary"):
            user = authenticate(username, password)
            if user:
                st.session_state["user"] = user
                st.success(f"Welcome, {user['username']}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with right:
        st.subheader("Register new user")
        new_user = st.text_input("New username")
        new_password = st.text_input("New password", type="password")
        role = st.selectbox("Role", ["analyst", "manager", "admin"], index=0)
        if st.button("Register"):
            ok = register_user(new_user, new_password, role=role)
            if ok:
                st.success("User registered. You can now log in.")
            else:
                st.error("Could not register user. The username may already exist.")

    st.info("Default demo login: username `admin`, password `admin123`.")


if "user" not in st.session_state:
    login_screen()
    st.stop()


user = st.session_state["user"]
with st.sidebar:
    st.markdown(f"### Logged in as: `{user['username']}`")
    st.markdown(f"Role: **{user['role']}**")
    if st.button("Logout"):
        st.session_state.pop("user", None)
        st.rerun()

    st.divider()
    page = st.radio(
        "Navigation",
        [
            "Executive Dashboard",
            "Data Explorer",
            "R&D and Game Performance",
            "Machine Learning Lab",
            "Device Health",
            "Anomaly Detection",
            "Data Quality Center",
            "Reports",
            "Azure / SAP Simulation",
            "Admin",
        ],
    )

    st.divider()
    if st.button("Regenerate synthetic demo data"):
        from src.generate_data import ensure_demo_data
        ensure_demo_data(force=True)
        reset_cache()
        st.success("Demo data regenerated.")
        st.rerun()


sessions, maintenance, devices, model_runs, features = load_data()
filtered_sessions = apply_filters(sessions)


# -----------------------------
# Page: Executive Dashboard
# -----------------------------
if page == "Executive Dashboard":
    st.title("Executive Dashboard")
    st.write("Stakeholder-oriented overview of synthetic entertainment-device telemetry.")

    kpis = executive_kpis(filtered_sessions)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Revenue", f"€{kpis['total_revenue']:,.0f}")
    c2.metric("Sessions", f"{kpis['total_sessions']:,}")
    c3.metric("Active Devices", f"{kpis['active_devices']:,}")
    c4.metric("Avg €/Session", f"€{kpis['avg_revenue_per_session']:.2f}")
    c5.metric("Error Rate", f"{kpis['error_rate']:.2%}")

    st.divider()

    if filtered_sessions.empty:
        st.warning("No data for selected filters.")
    else:
        daily_revenue = filtered_sessions.groupby("date", as_index=False)["revenue_eur"].sum()
        fig = px.line(daily_revenue, x="date", y="revenue_eur", title="Daily Revenue Trend")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            by_city = filtered_sessions.groupby("city", as_index=False)["revenue_eur"].sum().sort_values("revenue_eur", ascending=False)
            fig = px.bar(by_city, x="city", y="revenue_eur", title="Revenue by City")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            by_game = filtered_sessions.groupby("game_title", as_index=False)["revenue_eur"].sum().sort_values("revenue_eur", ascending=False)
            fig = px.bar(by_game, x="game_title", y="revenue_eur", title="Revenue by Game")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Simple business interpretation")
        top_game = by_game.iloc[0]["game_title"]
        top_city = by_city.iloc[0]["city"]
        st.success(
            f"The strongest revenue contribution currently comes from **{top_game}** and the strongest location is **{top_city}**. "
            "A product/R&D team could use this to compare game mechanics, software versions, and location patterns."
        )


# -----------------------------
# Page: Data Explorer
# -----------------------------
elif page == "Data Explorer":
    restrict_page(["admin", "analyst"])
    st.title("Data Explorer")
    st.write("Inspect joined telemetry data from the local SQL database.")

    st.subheader("Filtered session data")
    st.dataframe(filtered_sessions.head(1000), use_container_width=True)

    st.download_button(
        "Download filtered sessions as CSV",
        data=filtered_sessions.to_csv(index=False).encode("utf-8"),
        file_name="filtered_sessions.csv",
        mime="text/csv",
    )

    st.subheader("SQL Query Console")
    st.caption("For demo purposes, use SELECT queries only.")
    query = st.text_area("SQL query", value="SELECT game_title, COUNT(*) AS sessions, SUM(revenue_eur) AS revenue FROM game_sessions GROUP BY game_title ORDER BY revenue DESC;")
    if st.button("Run SQL query"):
        if not query.strip().lower().startswith("select"):
            st.error("Only SELECT queries are allowed in this demo console.")
        else:
            try:
                df_sql = read_sql(query)
                st.dataframe(df_sql, use_container_width=True)
            except Exception as e:
                st.error(f"SQL error: {e}")


# -----------------------------
# Page: R&D and Game Performance
# -----------------------------
elif page == "R&D and Game Performance":
    restrict_page(["admin", "analyst", "manager"])
    st.title("R&D and Game Performance")
    st.write("Analyze performance by game, category, software version, location, and device type.")

    if filtered_sessions.empty:
        st.warning("No data for selected filters.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["Game analysis", "Software version", "Device type", "Event logs"])

        with tab1:
            game_perf = filtered_sessions.groupby(["game_title", "game_category"], as_index=False).agg(
                sessions=("session_id", "count"),
                revenue=("revenue_eur", "sum"),
                avg_revenue=("revenue_eur", "mean"),
                errors=("error_count", "sum"),
                plays=("plays", "sum"),
            )
            game_perf["error_rate"] = game_perf["errors"] / game_perf["plays"].replace(0, np.nan)
            st.dataframe(game_perf.sort_values("revenue", ascending=False), use_container_width=True)
            fig = px.scatter(
                game_perf,
                x="sessions",
                y="revenue",
                size="avg_revenue",
                color="game_category",
                hover_name="game_title",
                title="Game Portfolio: Sessions vs Revenue",
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            sw_perf = filtered_sessions.groupby("software_version", as_index=False).agg(
                sessions=("session_id", "count"),
                revenue=("revenue_eur", "sum"),
                errors=("error_count", "sum"),
                plays=("plays", "sum"),
            )
            sw_perf["error_rate"] = sw_perf["errors"] / sw_perf["plays"].replace(0, np.nan)
            st.dataframe(sw_perf, use_container_width=True)
            fig = px.bar(sw_perf, x="software_version", y="revenue", color="error_rate", title="Software Version Revenue and Error Rate")
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            dev_perf = filtered_sessions.groupby("device_type", as_index=False).agg(
                sessions=("session_id", "count"),
                revenue=("revenue_eur", "sum"),
                errors=("error_count", "sum"),
                devices=("device_id", "nunique"),
            )
            st.dataframe(dev_perf, use_container_width=True)
            fig = px.bar(dev_perf, x="device_type", y="revenue", title="Revenue by Device Type")
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.write("Synthetic unstructured telemetry/event logs.")
            logs = filtered_sessions["event_log"].value_counts().reset_index()
            logs.columns = ["event_log", "count"]
            st.dataframe(logs, use_container_width=True)
            fig = px.bar(logs, x="count", y="event_log", orientation="h", title="Event Log Frequency")
            st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Page: Machine Learning Lab
# -----------------------------
elif page == "Machine Learning Lab":
    restrict_page(["admin", "analyst"])
    st.title("Machine Learning Lab")
    st.write("Train forecasting, classification, and anomaly-detection models.")

    st.subheader("Feature table")
    st.dataframe(features.head(1000), use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Revenue forecast", "Maintenance classifier", "Anomaly detector", "Model monitoring"])

    with tab1:
        st.markdown("### Revenue forecasting")
        st.write("Predict next-day revenue from daily device KPIs.")
        model_choice = st.selectbox("Forecasting model", ["Random Forest", "Linear Regression"])
        if st.button("Train revenue forecasting model"):
            try:
                model, metrics, pred_df, path = train_revenue_forecast(features, model_choice=model_choice)
                st.success(f"Model trained and saved to `{path}`.")
                st.json(metrics)
                st.dataframe(pred_df.head(100), use_container_width=True)
                fig = px.scatter(pred_df, x="actual_next_day_revenue", y="predicted_next_day_revenue", title="Actual vs Predicted Next-Day Revenue")
                st.plotly_chart(fig, use_container_width=True)
                reset_cache()
            except Exception as e:
                st.error(str(e))

    with tab2:
        st.markdown("### Maintenance-risk classifier")
        st.write("Predict whether a device is likely to require maintenance within the next 14 days.")
        clf_choice = st.selectbox("Classification model", ["Random Forest", "Logistic Regression"])
        if st.button("Train maintenance classifier"):
            try:
                model, metrics, pred_df, path = train_maintenance_classifier(features, model_choice=clf_choice)
                st.success(f"Model trained and saved to `{path}`.")
                st.json(metrics)
                st.dataframe(pred_df.sort_values("risk_probability", ascending=False).head(100) if "risk_probability" in pred_df else pred_df.head(100), use_container_width=True)
                if "risk_probability" in pred_df:
                    fig = px.histogram(pred_df, x="risk_probability", nbins=30, title="Predicted Maintenance Risk Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                reset_cache()
            except Exception as e:
                st.error(str(e))

    with tab3:
        st.markdown("### Anomaly detection")
        st.write("Use Isolation Forest to detect abnormal daily device behavior.")
        if st.button("Train anomaly detector"):
            try:
                model, metrics, anomaly_df, path = train_anomaly_detector(features)
                st.success(f"Anomaly detector trained and saved to `{path}`.")
                st.json(metrics)
                st.dataframe(anomaly_df.head(100), use_container_width=True)
                fig = px.scatter(
                    anomaly_df.head(500),
                    x="total_revenue",
                    y="total_errors",
                    color="is_anomaly",
                    hover_data=["device_id", "city", "software_version"],
                    title="Detected anomalies: Revenue vs Errors",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.session_state["anomaly_df"] = anomaly_df
                reset_cache()
            except Exception as e:
                st.error(str(e))

    with tab4:
        st.markdown("### Model run history")
        model_runs = read_sql("SELECT * FROM model_runs ORDER BY run_time DESC")
        st.dataframe(model_runs, use_container_width=True)
        if not model_runs.empty:
            fig = px.line(model_runs, x="run_time", y="metric_value", color="metric_name", facet_row="model_name", title="Model Metrics Over Runs")
            st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Page: Device Health
# -----------------------------
elif page == "Device Health":
    restrict_page(["admin", "analyst", "manager"])
    st.title("Device Health")
    st.write("Monitor error behavior, maintenance history, and device-level risk indicators.")

    if features.empty:
        st.warning("No feature data available.")
    else:
        latest_date = features["date"].max()
        latest = features[features["date"] == latest_date].copy()
        latest["risk_score_rule_based"] = (
            latest["error_rate"].fillna(0) * 55
            + (latest["total_errors"] / latest["total_errors"].max()) * 25
            + (latest["sessions"] / latest["sessions"].max()) * 20
        ).fillna(0)
        latest["risk_score_rule_based"] = latest["risk_score_rule_based"].clip(0, 100)

        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Highest risk devices")
            st.dataframe(
                latest.sort_values("risk_score_rule_based", ascending=False)[
                    ["device_id", "city", "device_type", "software_version", "sessions", "total_errors", "error_rate", "risk_score_rule_based"]
                ].head(20),
                use_container_width=True,
            )
        with c2:
            fig = px.histogram(latest, x="risk_score_rule_based", nbins=25, title="Rule-based Risk Score")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Maintenance logs")
        st.dataframe(maintenance.sort_values("maintenance_date", ascending=False).head(300), use_container_width=True)

        maint_count = maintenance.groupby("device_id", as_index=False).agg(maintenance_events=("maintenance_id", "count"))
        merged = latest.merge(maint_count, on="device_id", how="left").fillna({"maintenance_events": 0})
        fig = px.scatter(
            merged,
            x="total_errors",
            y="total_revenue",
            size="sessions",
            color="maintenance_events",
            hover_name="device_id",
            title="Device Health: Errors vs Revenue",
        )
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Page: Anomaly Detection
# -----------------------------
elif page == "Anomaly Detection":
    restrict_page(["admin", "analyst", "manager"])
    st.title("Anomaly Detection")
    st.write("Identify suspicious revenue drops, error spikes, and unusual telemetry patterns.")

    if "anomaly_df" not in st.session_state:
        try:
            _, metrics, anomaly_df, path = train_anomaly_detector(features)
            st.session_state["anomaly_df"] = anomaly_df
        except Exception:
            st.session_state["anomaly_df"] = pd.DataFrame()

    anomaly_df = st.session_state["anomaly_df"]
    if anomaly_df.empty:
        st.warning("No anomalies available. Train the anomaly detector in Machine Learning Lab.")
    else:
        anomalies = anomaly_df[anomaly_df["is_anomaly"] == 1].copy()
        st.metric("Detected anomalies", f"{len(anomalies):,}")
        st.dataframe(
            anomalies[
                ["date", "device_id", "city", "device_type", "software_version", "sessions", "total_revenue", "total_errors", "error_rate", "anomaly_score"]
            ].head(200),
            use_container_width=True,
        )
        fig = px.scatter(
            anomaly_df,
            x="total_revenue",
            y="total_errors",
            color="is_anomaly",
            hover_data=["date", "device_id", "city", "software_version"],
            title="Anomaly Map",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Stakeholder interpretation: anomalies are not automatically 'bad'. They are candidates for investigation, "
            "for example after a software update, location change, device issue, or unusual usage day."
        )


# -----------------------------
# Page: Data Quality Center
# -----------------------------
elif page == "Data Quality Center":
    restrict_page(["admin", "analyst"])
    st.title("Data Quality Center")
    st.write("Automated checks to show data-quality and governance awareness.")

    quality = data_quality_report(sessions)
    st.dataframe(quality, use_container_width=True)

    status_counts = quality["status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]
    fig = px.pie(status_counts, values="count", names="status", title="Data Quality Status")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Column completeness")
    completeness = (1 - sessions.isna().mean()).reset_index()
    completeness.columns = ["column", "completeness"]
    fig = px.bar(completeness, x="column", y="completeness", title="Column Completeness")
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Page: Reports
# -----------------------------
elif page == "Reports":
    restrict_page(["admin", "analyst", "manager"])
    st.title("Reports")
    st.write("Export stakeholder-ready reports.")

    quality = data_quality_report(sessions)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Create Excel report"):
            path = create_excel_report(filtered_sessions, features, quality)
            st.success(f"Excel report created: {path.name}")
            download_file_button(path, "Download Excel report", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with col2:
        if st.button("Create PDF summary"):
            try:
                path = create_pdf_summary(filtered_sessions, features, quality)
                st.success(f"PDF summary created: {path.name}")
                download_file_button(path, "Download PDF summary", "application/pdf")
            except Exception as e:
                st.error(f"Could not create PDF: {e}")

    st.subheader("Quick CSV exports")
    st.download_button(
        "Download filtered sessions CSV",
        data=filtered_sessions.to_csv(index=False).encode("utf-8"),
        file_name="filtered_sessions_export.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download daily device features CSV",
        data=features.to_csv(index=False).encode("utf-8"),
        file_name="daily_device_features_export.csv",
        mime="text/csv",
    )


# -----------------------------
# Page: Azure / SAP Simulation
# -----------------------------
elif page == "Azure / SAP Simulation":
    restrict_page(["admin", "analyst", "manager"])
    st.title("Azure / SAP Simulation")
    st.write("A portfolio-friendly simulation showing Azure Data and SAP Datasphere awareness.")

    tab1, tab2 = st.tabs(["Azure Data Pipeline Simulation", "SAP Datasphere Semantic Model"])

    with tab1:
        pipeline = simulate_azure_pipeline_status()
        st.dataframe(pipeline, use_container_width=True)
        fig = px.bar(pipeline, x="pipeline_name", y="records_processed", color="status", title="Pipeline Run Monitoring")
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            "Interview explanation: this simulates how telemetry data could move from device/event sources into a curated analytics layer, "
            "similar to Azure Data Factory or cloud data engineering workflows."
        )

    with tab2:
        semantic = simulate_sap_datasphere_model()
        st.dataframe(semantic, use_container_width=True)
        st.info(
            "This simulates a business semantic layer, where raw technical tables are converted into business-friendly objects "
            "such as Device, Location, Game Session, Revenue, Error Events, and Maintenance."
        )


# -----------------------------
# Page: Admin
# -----------------------------
elif page == "Admin":
    restrict_page(["admin"])
    st.title("Admin")
    st.write("Admin-only view for users, database counts, and app metadata.")

    st.subheader("Users")
    st.dataframe(pd.DataFrame(list_users()), use_container_width=True)

    st.subheader("Database tables")
    conn = sqlite3.connect(DB_PATH)
    table_names = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn)
    counts = []
    for t in table_names["name"]:
        try:
            count = pd.read_sql_query(f"SELECT COUNT(*) AS n FROM {t}", conn)["n"].iloc[0]
            counts.append({"table": t, "rows": int(count)})
        except Exception:
            pass
    conn.close()
    st.dataframe(pd.DataFrame(counts), use_container_width=True)

    st.subheader("Project metadata")
    st.markdown(
        """
        **Purpose:** Portfolio project for Data Scientist roles in R&D, gaming, entertainment-device analytics, and stakeholder reporting.  
        **Data:** Synthetic only.  
        **Core stack:** Python, Pandas, SQL/SQLite, scikit-learn, Streamlit, Plotly, Excel/PDF exports.  
        """
    )
