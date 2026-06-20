import sqlite3
import chromadb
from chromadb.utils import embedding_functions
import json


def build_vector_database():
    print("Connecting to the SQLite database...")
    conn = sqlite3.connect('batteries.db')
    cursor = conn.cursor()

    # Extract long-text fields that are useful for semantic retrieval.
    cursor.execute("SELECT id, battery_type, specific_material, mechanics FROM experiments WHERE mechanics IS NOT NULL")
    rows = cursor.fetchall()

    print(f"Found {len(rows)} text records. Preparing embeddings...")

    # Initialize the local Chroma vector database under chroma_db.
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # Use a lightweight open-source embedding model.
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        device="mps"
    )
    # Recreate the collection to avoid stale records.
    try:
        chroma_client.delete_collection(name="battery_mechanics")
        print("Existing vector collection removed.")
    except Exception:
        pass

    # Create a fresh collection.
    collection = chroma_client.create_collection(
        name="battery_mechanics",
        embedding_function=sentence_transformer_ef
    )

    documents = []
    metadatas = []
    ids = []

    for row in rows:
        exp_id, battery_type, material_raw, mechanics_raw = row

        # Parse JSON fields and assemble a compact semantic document.
        try:
            material_dict = json.loads(material_raw) if material_raw else {}
            mechanics_dict = json.loads(mechanics_raw) if mechanics_raw else {}

            cathode = material_dict.get('cathode', 'unknown cathode')
            dominant_mech = mechanics_dict.get('dominant_failure_mechanism', '')

            synonym_tags = ""
            if battery_type == "lfp":
                synonym_tags = "(aliases: LFP, LiFePO4, lithium iron phosphate)"
            elif battery_type == "ncm":
                synonym_tags = "(aliases: NCM, NMC, nickel cobalt manganese)"

            text = f"Material system: {cathode} {synonym_tags}. Failure mechanism and observed phenomena: {dominant_mech}"
        except Exception:
            text = f"Experimental material: {material_raw}. Failure mechanism and observed phenomena: {mechanics_raw}"

        documents.append(text)
        metadatas.append({
            "source": "sqlite_import",
            "exp_id": str(exp_id),
            "battery_type": battery_type if battery_type else "other"
        })
        ids.append(f"exp_{exp_id}")

    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print("Vector database built successfully under ./chroma_db.")
    else:
        print("No valid text records were found in the database.")


if __name__ == "__main__":
    build_vector_database()
