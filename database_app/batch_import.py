import sqlite3
import json
import re
from pathlib import Path

# Database path and JSON source folder.
DB_PATH = Path("batteries.db")
JSON_FOLDER = Path("json_data")


def extract_float(text):
    """Extract the first numeric value from a string."""
    if text is None or text == "":
        return None
    match = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", str(text))
    return float(match.group()) if match else None


def ensure_columns_exist(cursor):
    """Add missing columns for backward-compatible imports."""
    columns_to_add = {
        "observed_results_raw": "TEXT",
        "basic_info": "TEXT"
    }

    cursor.execute("PRAGMA table_info(experiments)")
    existing_columns = [row[1] for row in cursor.fetchall()]

    for col_name, col_type in columns_to_add.items():
        if col_name not in existing_columns:
            print(f"Adding missing column to experiments: {col_name}...")
            cursor.execute(f"ALTER TABLE experiments ADD COLUMN {col_name} {col_type}")


def normalize_boundary_and_thermal(group):
    """
    Normalize legacy and current schemas into boundary_and_thermal.
    Legacy format: boundary_and_thermal.
    Current format: test_setup + thermal_conditions + mitigation_strategy.
    """
    old_data = group.get("boundary_and_thermal")
    if isinstance(old_data, dict) and old_data:
        return old_data

    merged = {}
    test_setup = group.get("test_setup", {})
    thermal_conditions = group.get("thermal_conditions", {})

    if isinstance(test_setup, dict):
        merged.update(test_setup)

    if isinstance(thermal_conditions, dict):
        merged.update(thermal_conditions)

    if "mitigation_strategy" in group:
        merged["mitigation_strategy"] = group.get("mitigation_strategy")

    return merged


def normalize_mechanics(group):
    """
    Normalize legacy and current schemas into mechanics.
    Current files may omit mechanics, in which case an empty object is used.
    """
    mechanics = group.get("mechanics")
    if isinstance(mechanics, dict):
        return mechanics
    return {}


def normalize_basic_info(group):
    """
    Normalize basic experiment metadata.
    """
    return {
        "trigger_method": group.get("trigger_method"),
        "cell_number": group.get("cell_number"),
        "ambient_pressure": group.get("ambient_pressure"),
        "tr_occurred": group.get("tr_occurred")
    }


def detect_battery_type(group):
    """
    Detect battery system from the cathode field.
    """
    mat = group.get("specific_material", {}) or {}
    cathode = str(mat.get("cathode", "")).lower()

    if any(k in cathode for k in ["ncm", "nmc", "nickel cobalt manganese"]):
        return "ncm"
    elif any(k in cathode for k in ["lfp", "lifepo4", "lithium iron phosphate"]):
        return "lfp"
    elif any(k in cathode for k in ["na-ion", "sodium"]):
        return "na"
    elif any(k in cathode for k in ["lco", "licoo2", "lithium cobalt oxide"]):
        return "lco"
    elif any(k in cathode for k in ["lmo", "limn2o4", "lithium manganese oxide"]):
        return "lmo"
    elif any(k in cathode for k in ["nca", "nickel cobalt aluminum"]):
        return "nca"
    else:
        return "other"


def batch_import():
    if not JSON_FOLDER.exists():
        print(f"JSON folder not found: {JSON_FOLDER}. Create it and add JSON files first.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        ensure_columns_exist(cursor)
        conn.commit()
    except Exception as e:
        print(f"Schema check failed. If the tables do not exist yet, initialize the database first: {e}")

    success_count = 0
    error_count = 0

    for json_file in JSON_FOLDER.glob("*.json"):
        print(f"Reading: {json_file.name} ...", end=" ")

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Top-level compatibility.
            paper_id = data.get("paper_id") or data.get("paper_doi_url") or "Unknown_DOI"
            groups = data.get("experimental_groups") or data.get("groups") or []

            # Write paper metadata.
            cursor.execute(
                """
                INSERT OR IGNORE INTO papers (paper_id, title, publish_year)
                VALUES (?, ?, ?)
                """,
                (paper_id, data.get("title", "Title pending"), data.get("publish_year", 2025))
            )

            imported_types = set()

            for group in groups:
                mat = group.get("specific_material", {}) or {}
                obs = group.get("observed_results", {}) or {}

                b_type = detect_battery_type(group)
                imported_types.add(b_type.upper())

                basic_info = normalize_basic_info(group)
                boundary_and_thermal = normalize_boundary_and_thermal(group)
                mechanics = normalize_mechanics(group)

                cursor.execute(
                    """
                    INSERT INTO experiments (
                        paper_id, battery_type, group_id,
                        T_onset, T_trigger, max_temperature, time_to_TR,
                        specific_material, boundary_and_thermal, mechanics,
                        observed_results_raw, basic_info
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        paper_id,
                        b_type,
                        group.get("group_id"),
                        extract_float(obs.get("T_onset")),
                        extract_float(obs.get("T_trigger")),
                        extract_float(obs.get("max_temperature")),
                        extract_float(obs.get("time_to_TR")),
                        json.dumps(mat, ensure_ascii=False),
                        json.dumps(boundary_and_thermal, ensure_ascii=False),
                        json.dumps(mechanics, ensure_ascii=False),
                        json.dumps(obs, ensure_ascii=False),
                        json.dumps(basic_info, ensure_ascii=False),
                    )
                )

            conn.commit()
            type_text = ", ".join(sorted(imported_types)) if imported_types else "UNKNOWN"
            print(f"Success ({len(groups)} experiment groups imported; detected systems: {type_text})")
            success_count += 1

        except Exception as e:
            conn.rollback()
            print(f"Failed: {e}")
            error_count += 1

    conn.close()
    print("-" * 40)
    print(f"Batch import complete. Success: {success_count} papers; failed: {error_count} papers.")


if __name__ == "__main__":
    batch_import()
