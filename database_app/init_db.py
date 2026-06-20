import sqlite3
import json
from pathlib import Path

DB_PATH = Path("batteries.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Table 1: paper-level metadata.
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS papers
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       paper_id
                       TEXT
                       UNIQUE
                       NOT
                       NULL,
                       title
                       TEXT,
                       publish_year
                       INTEGER
                   )''')

    # Table 2: experiment-group records with scalar fields and nested JSON.
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS experiments
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       paper_id
                       TEXT
                       NOT
                       NULL,
                       battery_type
                       TEXT
                       NOT
                       NULL, -- 'ncm', 'lfp', 'other'
                       group_id
                       TEXT
                       NOT
                       NULL,
                       T_onset
                       REAL,
                       T_trigger
                       REAL,
                       max_temperature
                       REAL,
                       time_to_TR
                       REAL,
                       specific_material
                       TEXT, -- JSON payload
                       boundary_and_thermal
                       TEXT, -- JSON payload
                       mechanics
                       TEXT, -- JSON payload
                       FOREIGN
                       KEY
                   (
                       paper_id
                   ) REFERENCES papers
                   (
                       paper_id
                   ) ON DELETE CASCADE
                       )''')

    conn.commit()
    conn.close()
    print("Database schema initialized.")


if __name__ == "__main__":
    init_db()
