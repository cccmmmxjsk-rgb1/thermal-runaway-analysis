from flask import Flask, request, jsonify, render_template, Response
import requests
import sqlite3
import json
from pathlib import Path
import chromadb


app = Flask(__name__)
DB_PATH = Path("batteries.db")

# ================= Vector database initialization =================
chroma_client = chromadb.PersistentClient(path="./chroma_db")
try:
    vector_collection = chroma_client.get_collection("battery_mechanics")
except:
    vector_collection = None


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = 1")
    return conn


# ================= Page routes =================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/battery")
def battery_page():
    return render_template("battery.html")


@app.route("/ai_workspace")
def ai_workspace_page():
    return render_template("ai_workspace.html")


# ================= API routes: nested data query =================
@app.route("/api/papers/<battery_type>", methods=["GET"])
def get_papers_with_experiments(battery_type):
    conn = get_conn()
    where_clause = ""
    params = ()
    if battery_type != "all":
        where_clause = "WHERE e.battery_type = ?"
        params = (battery_type,)

    query = f"""
            SELECT p.paper_id, e.id as exp_id, e.group_id, e.battery_type, e.T_onset, 
                   e.T_trigger, e.max_temperature, e.time_to_TR, e.specific_material, 
                   e.boundary_and_thermal, e.mechanics, e.observed_results_raw, e.basic_info
            FROM papers p
            JOIN experiments e ON p.paper_id = e.paper_id
            {where_clause}
            ORDER BY p.id DESC, e.id ASC 
            """
    rows = conn.execute(query, params).fetchall()
    conn.close()

    result_map = {}
    for row in rows:
        pid = row['paper_id']
        if pid not in result_map:
            result_map[pid] = {"paper_id": pid, "experiments": []}
        exp = {
            "exp_id": row['exp_id'], "group_id": row['group_id'], "battery_type": row['battery_type'],
            "T_onset": row['T_onset'], "T_trigger": row['T_trigger'], "max_temperature": row['max_temperature'],
            "time_to_TR": row['time_to_TR'],
            "specific_material": json.loads(row['specific_material'] or '{}'),
            "boundary_and_thermal": json.loads(row['boundary_and_thermal'] or '{}'),
            "mechanics": json.loads(row['mechanics'] or '{}'),
            "observed_results_raw": json.loads(row['observed_results_raw'] or '{}'),
            "basic_info": json.loads(row['basic_info'] or '{}')
        }
        result_map[pid]['experiments'].append(exp)
    return jsonify(list(result_map.values()))


# ================= API routes: data deletion =================
@app.route("/api/delete_data", methods=["POST"])
def delete_data():
    data = request.get_json()
    del_type = data.get("type")
    target_id = data.get("id")

    conn = get_conn()
    try:
        if del_type == "paper":
            # Collect experiment IDs so ChromaDB can be synchronized.
            rows = conn.execute("SELECT id FROM experiments WHERE paper_id = ?", (target_id,)).fetchall()
            exp_ids = [f"exp_{r['id']}" for r in rows]

            # Delete from SQLite.
            conn.execute("DELETE FROM experiments WHERE paper_id = ?", (target_id,))
            conn.execute("DELETE FROM papers WHERE paper_id = ?", (target_id,))

            # Synchronize ChromaDB.
            if vector_collection and exp_ids:
                try:
                    vector_collection.delete(ids=exp_ids)
                except Exception as ve:
                    print(f"Failed to clean paper vectors: {ve}")

        elif del_type == "experiment":
            # Delete a single experiment group.
            conn.execute("DELETE FROM experiments WHERE id = ?", (target_id,))

            if vector_collection:
                try:
                    vector_collection.delete(ids=[f"exp_{target_id}"])
                except Exception as ve:
                    print(f"Failed to clean experiment vector: {ve}")

        conn.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()
# ================= API routes: AI model dispatch =================
@app.route("/api/chat", methods=["POST"])
def chat_with_model():
    data = request.get_json()
    prompt = data.get("prompt", "")

    # Normalize common battery-material aliases before retrieval.
    synonyms = {
        "lifepo4": "LFP",
        "lithium iron phosphate": "LFP",
        "nmc": "NCM",
        "sodium-ion": "na",
        "sodium ion": "na",
        "lithium cobalt oxide": "lco",
        "lithium manganese oxide": "lmo",
        "nickel cobalt aluminum": "nca"
    }

    processed_prompt = prompt
    for k, v in synonyms.items():
        import re
        processed_prompt = re.sub(k, v, processed_prompt, flags=re.IGNORECASE)

    model_type = data.get("modelType", "ollama")
    selected_dbs = data.get("dbs", [])
    retrieval_mode = data.get("retrievalMode", "sqlite")
    history = data.get("chatHistory", [])
    evaluate_safety = data.get("evaluateSafety", False)
    enable_hybrid_rag = data.get("enableHybridRag", True)

    context_text = ""

    # ================= 1. Retrieval stage =================
    if retrieval_mode == "sqlite":
        context_data = []
        conn = get_conn()
        for db in selected_dbs:
            if db.startswith("battery_"):
                b_type = db.split("_")[1]
                rows = conn.execute("SELECT group_id, T_trigger, mechanics FROM experiments WHERE battery_type = ?",
                                    (b_type,)).fetchall()
                for r in rows:
                    t_trig = f"Trigger temperature: {r['T_trigger']}°C" if r['T_trigger'] else "Trigger temperature: missing"
                    context_data.append(f"[{b_type.upper()} - {r['group_id']}] {t_trig}\nFailure mechanism: {r['mechanics']}")
        conn.close()
        if context_data:
            context_text = "[Data retrieved by SQLite exact query]\n" + "\n\n".join(context_data)

    elif retrieval_mode == "vector":
        if vector_collection:
            b_types = [db.split("_")[1] for db in selected_dbs if db.startswith("battery_")]
            where_clause = {"battery_type": {"$in": b_types}} if b_types else None

            results = vector_collection.query(
                query_texts=[prompt],
                n_results=100,
                where=where_clause
            )
            if results and results.get('documents') and results['documents'][0]:
                context_text = "[Relevant snippets retrieved from Chroma vector search]\n" + "\n\n".join(results['documents'][0])
        else:
            context_text = "[Notice] The vector database is not initialized. Please run build_vector_db.py first."



    elif retrieval_mode == "graph":
        conn = get_conn()
        b_types = [db.split("_")[1] for db in selected_dbs if db.startswith("battery_")]
        if not b_types:
            b_types = ['ncm', 'lfp', 'other']
        placeholders = ','.join('?' * len(b_types))
        # Include e.id as exp_id for vector-pruned GraphRAG.
        query = f"""
                SELECT p.paper_id, e.id as exp_id, e.group_id, e.battery_type, e.T_trigger, e.max_temperature 
                FROM papers p 
                JOIN experiments e ON p.paper_id = e.paper_id 
                WHERE e.battery_type IN ({placeholders})
            """
        rows = conn.execute(query, b_types).fetchall()
        conn.close()
        filtered_rows = []
        if enable_hybrid_rag and vector_collection and prompt.strip():
            # Route A: semantic pruning with Hybrid RAG.
            try:
                where_clause = {"battery_type": {"$in": b_types}} if b_types else None
                vector_results = vector_collection.query(
                    query_texts=[prompt],
                    n_results=1000,
                    where=where_clause
                )
                if vector_results and vector_results.get('ids') and vector_results['ids'][0]:
                    relevant_exp_ids = [int(vid.replace("exp_", "")) for vid in vector_results['ids'][0]]
                    filtered_rows = [r for r in rows if r['exp_id'] in relevant_exp_ids]
            except Exception as e:
                print(f"Semantic pruning failed: {e}")
        else:
            # Route B: simple keyword heuristic.
            user_prompt_lower = prompt.lower()
            for r in rows:
                row_text = f"{r['paper_id']} {r['group_id']} {r['T_trigger']} {r['max_temperature']} {r['battery_type']}".lower()
                if len(user_prompt_lower) > 5 and any(
                        word in row_text for word in user_prompt_lower.split() if len(word) > 1):
                    filtered_rows.append(r)
        # Fallback: cap the graph context to avoid flooding the model.
        if not filtered_rows:
            filtered_rows = rows[:1000]
        mode_text = "semantic vector pruning enabled" if enable_hybrid_rag else "rule-based pruning enabled"
        graph_lines = [f"[GraphRAG entity-relation network: {mode_text}]", "Path schema: (Material) -> (Paper) -> (Experiment Group)"]
        paper_map = {}
        for r in filtered_rows:
            pid = r['paper_id']
            b_type = r['battery_type'].upper()
            if pid not in paper_map:
                paper_map[pid] = {"material": b_type, "exps": []}
            paper_map[pid]["exps"].append(
                {"group_id": r['group_id'], "T_trigger": r['T_trigger'], "T_max": r['max_temperature']})
        for pid, info in paper_map.items():
            graph_lines.append(f"(Material system: {info['material']}) --[reported in]--> (Paper: {pid})")
            for exp in info["exps"]:
                t_str = f"{exp['T_trigger']}℃" if exp['T_trigger'] else "unknown"
                m_str = f"{exp['T_max']}℃" if exp['T_max'] else "unknown"
                graph_lines.append(
                    f"  └─ (Paper: {pid}) --[contains]--> (Group: {exp['group_id']}) | T_trigger={t_str}, T_max={m_str}")
        if len(graph_lines) > 2:
            context_text = "\n".join(graph_lines)

    # ================= 2. Build the message payload =================
    base_system_prompt = "You are a senior materials science and AI-for-science expert. Answer in English."
    if retrieval_mode == "graph" and context_text:
        base_system_prompt += f"Use the provided entity-relation chains for careful multi-hop reasoning. Do not mix facts across papers or experiment groups.\n\n{context_text}"
    elif context_text:
        base_system_prompt += f"Base your comparison and reasoning strictly on the extracted literature knowledge below:\n\n{context_text}"
    else:
        base_system_prompt += "No knowledge collection was selected. Answer directly and clearly."

    if evaluate_safety:
        base_system_prompt += "\n\nRequired constraint: include the trade-off between high specific energy and intrinsic safety. Provide an integrated assessment of safety and energy density for the retrieved systems, and use it to guide safer battery-system design."
    messages = [{"role": "system", "content": base_system_prompt}]
    for msg in history:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    messages.append({"role": "user", "content": prompt})

    # ================= 3. Model routing with streaming =================
    if model_type == "ollama":
        model_name = data.get("modelName", "llama3")
        try:
            response = requests.post("http://localhost:11434/api/chat",
                                     json={"model": model_name, "messages": messages, "stream": True}, stream=True)
            response.raise_for_status()

            def generate_ollama():
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk_data = json.loads(line.decode('utf-8'))
                            if "message" in chunk_data and "content" in chunk_data["message"]:
                                yield chunk_data["message"]["content"]
                        except json.JSONDecodeError:
                            pass

            return Response(generate_ollama(), content_type='text/plain; charset=utf-8')
        except Exception as e:
            return jsonify({"error": f"Ollama call failed: {str(e)}"}), 500

    elif model_type == "api":
        api_key = data.get("apiKey", "")
        base_url = data.get("apiBaseUrl", "https://api.deepseek.com").rstrip("/")
        model_name = data.get("apiModelName", "deepseek-chat")

        if not api_key: return jsonify({"error": "An API key is required in cloud API mode."}), 400
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        try:
            resp = requests.post(f"{base_url}/chat/completions", headers=headers,
                                 json={"model": model_name, "messages": messages, "stream": True},
                                 stream=True, timeout=60)

            if resp.status_code != 200:
                return jsonify({"error": f"API rejected the request, status {resp.status_code}: {resp.text}"}), 500

            def generate_api():
                for line in resp.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            data_str = decoded_line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data_json = json.loads(data_str)
                                delta = data_json["choices"][0]["delta"]
                                if "content" in delta:
                                    yield delta["content"]
                            except json.JSONDecodeError:
                                pass

            return Response(generate_api(), content_type='text/plain; charset=utf-8')

        except Exception as e:
            return jsonify({"error": f"Cloud API call failed or network connection failed: {str(e)}"}), 500

    return jsonify({"error": "Unknown model type."}), 400


@app.route("/api/graph_data", methods=["GET"])
def get_graph_data():
    conn = get_conn()
    query = "SELECT p.paper_id, e.group_id, e.battery_type FROM papers p JOIN experiments e ON p.paper_id = e.paper_id"
    rows = conn.execute(query).fetchall()
    conn.close()

    nodes_dict = {}
    links = []
    added_links = set()

    cathode_mapping = {
        'ncm': 'NCM',
        'lfp': 'LFP',
        'na': 'Na-ion',
        'lco': 'LCO',
        'lmo': 'LMO',
        'nca': 'NCA'
    }
    cat_mapping = {'ncm': 1, 'lfp': 2, 'na': 3, 'lco': 4, 'lmo': 5, 'nca': 6}

    for row in rows:
        paper_id = row["paper_id"]
        group_id = row["group_id"]
        b_type = row["battery_type"].lower() if row["battery_type"] else "other"

        cathode_name = cathode_mapping.get(b_type, 'Other')
        sys_cat = cat_mapping.get(b_type, 7)

        short_paper_name = paper_id.split('/')[-1] if '/' in paper_id else paper_id

        mat_node_id = f"mat_{b_type}"
        exp_node_id = f"exp_{group_id}"

        if mat_node_id not in nodes_dict:
            nodes_dict[mat_node_id] = {"id": mat_node_id, "name": f"🧪 {cathode_name}", "category": sys_cat, "nodeType": "material"}

        if paper_id not in nodes_dict:
            nodes_dict[paper_id] = {"id": paper_id, "name": f"📄 {short_paper_name}", "category": sys_cat, "nodeType": "paper"}

        mat_paper_link = f"{mat_node_id}->{paper_id}"
        if mat_paper_link not in added_links:
            links.append({"source": mat_node_id, "target": paper_id, "value": "reported in"})
            added_links.add(mat_paper_link)

        if exp_node_id not in nodes_dict:
            nodes_dict[exp_node_id] = {"id": exp_node_id, "name": group_id, "category": sys_cat, "nodeType": "group", "parentId": paper_id}

        paper_exp_link = f"{paper_id}->{exp_node_id}"
        if paper_exp_link not in added_links:
            links.append({"source": paper_id, "target": exp_node_id, "value": "contains experiment"})
            added_links.add(paper_exp_link)

    categories = [
        {"name": "Unused"},
        {"name": "NCM System"},
        {"name": "LFP System"},
        {"name": "Na-ion System"},
        {"name": "LCO System"},
        {"name": "LMO System"},
        {"name": "NCA System"},
        {"name": "Other Systems"}
    ]
    return jsonify({"nodes": list(nodes_dict.values()), "links": links, "categories": categories})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
