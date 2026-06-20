# Battery Thermal Runaway Database

An open, community-oriented database and analysis workspace for battery thermal runaway literature. The project combines structured experiment records, paper-to-experiment browsing, a knowledge graph view, and AI-assisted retrieval workflows.

## Features

- Structured SQLite database for papers and experiment groups.
- Battery-system tabs for NCM, LFP, Na-ion, LCO, LMO, NCA, and other systems.
- Nested paper-to-experiment tables with global search and detail drawers.
- ECharts knowledge graph for material-paper-experiment relationships.
- AI analysis workspace with SQLite RAG, ChromaDB vector retrieval, and GraphRAG-style context assembly.
- Batch import pipeline for extracted literature JSON files.
- Local vector database builder for mechanism-focused semantic retrieval.

## Project Layout

- `app.py`: Flask application, API routes, retrieval logic, graph data, and AI model dispatch.
- `templates/`: HTML templates for the public hub, database browser, and AI workspace.
- `init_db.py`: SQLite schema initialization.
- `batch_import.py`: Batch import from `json_data/*.json` into `batteries.db`.
- `build_vector_db.py`: ChromaDB vector database builder.
- `draw_real_data_sankey.py`: Plotly Sankey chart for cathode-name normalization.
- `json_data/`: Extracted literature records used as source data.
- `batteries.db`: SQLite database generated from imported literature records.
- `chroma_db/`: Persistent ChromaDB vector store.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Initialize and Import Data

```bash
python init_db.py
python batch_import.py
python build_vector_db.py
```

## Run the App

```bash
python app.py
```

Then open:

```text
http://localhost:5001
```

## AI Workspace

The AI workspace supports:

- Local Ollama models through `http://localhost:11434/api/chat`.
- OpenAI-compatible cloud APIs by entering an API key, base URL, and model name in the UI.

No API key is stored in the repository. Users must provide their own key at runtime.

## Public Release Notes

Before publishing this repository publicly, verify redistribution rights for:

- Source literature metadata and extracted JSON files.
- The bundled SQLite database.
- Any archived files or derived datasets.

If redistribution is not fully cleared, publish the application code separately and provide a reproducible import workflow for users who have lawful access to the source documents.
