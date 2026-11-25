"""
Database engine and session management for the Disaster Response Backend.
Uses SQLite for persistent storage of real SOS alerts and incidents.
"""

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "disaster_alerts.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def ensure_schema_migrations():
    """Ensures newly added columns exist in SQLite tables without data loss."""
    import sqlite3
    con = sqlite3.connect(str(DB_PATH))
    cur = con.cursor()
    try:
        existing_cols = [col[1] for col in cur.execute("PRAGMA table_info(emergency_sos_records)").fetchall()]
        if existing_cols:
            new_cols = [
                ("assigned_team_id", "TEXT"),
                ("assigned_team_name", "TEXT"),
                ("team_distance_km", "REAL"),
                ("assignment_status", "TEXT DEFAULT 'Awaiting Rescue Team Assignment'"),
                ("estimated_eta_minutes", "INTEGER"),
            ]
            for col_name, col_type in new_cols:
                if col_name not in existing_cols:
                    cur.execute(f"ALTER TABLE emergency_sos_records ADD COLUMN {col_name} {col_type}")
            con.commit()
    except Exception as e:
        print(f"Migration note: {e}")
    finally:
        con.close()


def get_db():
    """Dependency that yields a database session and closes it afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
