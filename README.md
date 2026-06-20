# thermal-runaway-analysis

Python code and source data for battery thermal runaway analysis.

This repository contains two related parts:

- `scripts/`: plotting and analysis scripts used to generate figures from curated Excel datasets.
- `database_app/`: a Flask-based literature database and AI-assisted analysis workspace built from extracted thermal-runaway literature records.

## Plotting Scripts

The plotting scripts are under `scripts/visualization/`, with source spreadsheets under `scripts/data/`.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a figure script from the visualization directory, for example:

```bash
cd scripts/visualization
python 2a.py
```

## Database App

The database app is under `database_app/`. It includes:

- extracted JSON records in `database_app/json_data/`
- a generated SQLite database, `database_app/batteries.db`
- a small ChromaDB vector index under `database_app/chroma_db/`
- Flask pages for browsing papers, experiments, graph data, and AI-assisted retrieval

Run it locally:

```bash
cd database_app
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://localhost:5001
```

To rebuild the database from JSON records:

```bash
cd database_app
python init_db.py
python batch_import.py
python build_vector_db.py
```

## Data Notice

The repository contains derived literature records and analysis datasets. Before redistribution or reuse, verify that the source literature metadata, extracted records, and derived databases can be shared under the intended license and use case.
