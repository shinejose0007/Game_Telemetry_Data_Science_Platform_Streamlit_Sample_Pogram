from pathlib import Path
import pandas as pd
from datetime import datetime
from fpdf import FPDF

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


def create_excel_report(sessions: pd.DataFrame, features: pd.DataFrame, quality: pd.DataFrame) -> Path:
    path = REPORT_DIR / f"stakeholder_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        sessions.head(2000).to_excel(writer, sheet_name="Raw Sessions Sample", index=False)
        features.head(2000).to_excel(writer, sheet_name="Daily Device KPIs", index=False)
        quality.to_excel(writer, sheet_name="Data Quality", index=False)

        if not sessions.empty:
            by_game = sessions.groupby("game_title", as_index=False).agg(
                revenue_eur=("revenue_eur", "sum"),
                sessions=("session_id", "count"),
                errors=("error_count", "sum"),
            ).sort_values("revenue_eur", ascending=False)
            by_game.to_excel(writer, sheet_name="Game Performance", index=False)

    return path


def create_pdf_summary(sessions: pd.DataFrame, features: pd.DataFrame, quality: pd.DataFrame) -> Path:
    path = REPORT_DIR / f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Game Telemetry Data Science Platform", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 8, "Executive stakeholder summary generated from synthetic entertainment-device telemetry data.")
    pdf.ln(3)

    if not sessions.empty:
        total_revenue = sessions["revenue_eur"].sum()
        total_sessions = len(sessions)
        active_devices = sessions["device_id"].nunique()
        total_errors = sessions["error_count"].sum()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Key Performance Indicators", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, f"Total revenue: EUR {total_revenue:,.2f}", ln=True)
        pdf.cell(0, 7, f"Total sessions: {total_sessions:,}", ln=True)
        pdf.cell(0, 7, f"Active devices: {active_devices:,}", ln=True)
        pdf.cell(0, 7, f"Total error events: {total_errors:,}", ln=True)
        pdf.ln(3)

        top_games = sessions.groupby("game_title")["revenue_eur"].sum().sort_values(ascending=False).head(5)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Top 5 Games by Revenue", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for game, revenue in top_games.items():
            pdf.cell(0, 7, f"{game}: EUR {revenue:,.2f}", ln=True)
        pdf.ln(3)

    if quality is not None and not quality.empty:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Data Quality Summary", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for _, row in quality.iterrows():
            pdf.multi_cell(0, 7, f"{row['check']} - {row['status']}: {row['details']}")

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(
        0,
        6,
        "Note: This report is based only on synthetic demo data and is intended for portfolio.",
    )
    pdf.output(str(path))
    return path
