from src.generate_data import ensure_demo_data

if __name__ == "__main__":
    ensure_demo_data(force=True)
    print("Demo data created in data/game_telemetry.db and CSV files in /data.")
