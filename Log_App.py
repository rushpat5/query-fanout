# app.py
"""
Qforia-free (AIO-style approximation) - REST-only Flash models.
Produces query fan-outs conditioned on pasted SERP context to approximate Google AI Overview outputs.
"""

import streamlit as st
import requests
import json
import re
import pandas as pd
from typing import List, Dict, Any, Optional
import difflib

# Optional: use python-Levenshtein if available for faster ratios
try:
    import Levenshtein
    def similarity(a: str, b: str) -> float:
        if not a or not b: return 0.0
        try:
            return Levenshtein.ratio(a, b)
        except Exception:
            return difflib.SequenceMatcher(None, a, b).ratio()
except Exception:
    def similarity(a: str, b: str) -> float:
        if not a or not b: return 0.0
        return difflib.SequenceMatcher(None, a, b).ratio()

# ---------- Config ----------
FLASH_MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-flash"
]
BASE_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# ---------- Helpers ----------
def build_prompt(seed: str, region: str, n: int, serp_context: str, extra: str) -> str:
    ts = pd.Timestamp.utcnow().isoformat() + "Z"
    context_block = ""
    if serp_context and serp_context.strip():
        # limit length of context shown
        ctx = serp_context.strip()
        if len(ctx) > 3000:
            ctx = ctx[:3000] + "\n...TRUNCATED..."
        context_block = f"\nSERP context (titles/snippets/PAA/related):\n{ctx}\n"
    prompt = f"""
You are simulating the types of user queries an AI Overview (search summary) would surface for a seed query.
Produce EXACTLY {n} items as a JSON array. Each item must be an object with fields:
 - query (string)
 - intent (string): one of informational, commercial, transactional, navigational, investigational
 - entity (string): main entity or brand (or empty string)
 - variation_type (string): paraphrase, narrowing, expansion, entity-focus, question-form, long-tail, comparative
 - region_weight (number 0-1): how strongly this variant is tied to the region
 - rationale (string): one-sentence reason why this query is valuable or distinct

Seed query: "{seed}"
Region hint: "{region}"
Timestamp: {ts}
{context_block}

Important requirements:
 - Output ONLY a JSON array and NOTHING else (no markdown, no commentary).
 - Bias wording and spelling to the provided region.
 - Favor queries that reflect typical SERP signals: questions, how-tos, comparisons, local intent, and named entities.
 - If the SERP context contains titles or PAA entries, prefer generating queries that are consistent with those signals.
 - If you cannot produce all fields naturally, set entity to empty string, region_weight to 0.0, and rationale to a short explanation.

Extra instructions:
{extra}
"""
    return prompt.strip()

def call_model_rest(api_key: str, model: str, prompt: str, timeout: int = 40) -> Dict[str, Any]:
    url = BASE_TEMPLATE.format(model=model)
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(url, params={"key": api_key}, json=payload, timeout=timeout)
        return {"ok": r.status_code == 200, "status": r.status_code, "body": r.text, "url": url}
    except Exception as e:
        return {"ok": False, "status": None, "body": str(e), "url": url}

def extract_text_from_response_body(body: str) -> Optional[str]:
    try:
        data = json.loads(body)
    except Exception:
        return None
    try:
        cand = data.get("candidates")
        if cand and isinstance(cand, list) and len(cand) > 0:
            content = cand[0].get("content")
            if content and isinstance(content, dict):
                parts = content.get("parts")
                if parts and isinstance(parts, list):
                    texts = [p.get("text","") for p in parts if isinstance(p, dict)]
                    return "".join(texts)
    except Exception:
        pass
    # fallback
    for k in ("output","text","content"):
        if k in data and isinstance(data[k], str):
            return data[k]
    return None

def strip_code_fence(text: str) -> str:
    if not text:
        return text
    t = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()

def parse_json_array_from_text(text: str) -> Optional[List[Any]]:
    if text is None:
        return None
    t = strip_code_fence(text)
    try:
        parsed = json.loads(t)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    m = re.search(r"\[[\s\S]*\]", t)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return None
    return None

def tokenize_for_overlap(text: str) -> List[str]:
    if not text:
        return []
    s = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = [t for t in s.split() if len(t) > 2]
    return tokens

def serp_signals_from_context(serp_context: str) -> List[str]:
    # naive: split lines and keep non-empty short lines
    if not serp_context:
        return []
    lines = [l.strip() for l in serp_context.splitlines() if l.strip()]
    # prefer lines shorter than ~200 chars
    lines = [l for l in lines if len(l) < 400]
    # take top 50 lines
    return lines[:50]

def compute_serp_overlap_score(query: str, serp_tokens: List[str]) -> float:
    if not serp_tokens:
        return 0.0
    q_tokens = set(tokenize_for_overlap(query))
    if not q_tokens:
        return 0.0
    serp_token_set = set(serp_tokens)
    overlap = q_tokens & serp_token_set
    return len(overlap) / max(1, len(q_tokens))

def compute_best_serp_similarity(query: str, serp_lines: List[str]) -> float:
    best = 0.0
    for line in serp_lines:
        sim = similarity(query, line)
        if sim > best:
            best = sim
    return best

def normalize_item(item: Any) -> Dict[str, Any]:
    # If item is a string, convert to object with defaults
    if isinstance(item, str):
        return {"query": item, "intent": "", "entity": "", "variation_type": "", "region_weight": 0.0, "rationale": ""}
    if not isinstance(item, dict):
        return {"query": str(item), "intent": "", "entity": "", "variation_type": "", "region_weight": 0.0, "rationale": ""}
    return {
        "query": item.get("query",""),
        "intent": item.get("intent",""),
        "entity": item.get("entity",""),
        "variation_type": item.get("variation_type",""),
        "region_weight": float(item.get("region_weight", 0.0)) if item.get("region_weight") is not None else 0.0,
        "rationale": item.get("rationale","")
    }

# --------------------------- Streamlit UI ---------------------------
st.set_page_config(layout="wide", page_title="Qforia - AIO-style (Flash + SERP context)")
st.title("Qforia — AIO-style Query Fan-Out (Flash, REST-only)")

with st.sidebar:
    st.header("Authentication")
    api_key = st.text_input("Gemini API Key (AI Studio, free-tier)", type="password")
    st.markdown("---")
    st.write("Flash model candidates (order matters). The app will try them in sequence until one succeeds.")
    models_selected = st.multiselect("Models", FLASH_MODEL_CANDIDATES, default=FLASH_MODEL_CANDIDATES[:2])
    if not models_selected:
        models_selected = FLASH_MODEL_CANDIDATES[:2]
    st.markdown("---")
    st.write("Optional: upload Search Console CSV (columns: query, clicks, impressions) to score/validate outputs")
    sc_file = st.file_uploader("Search Console CSV (optional)", type=["csv"])

st.subheader("Fan-out configuration")
seed_query = st.text_input("Seed query", value="how to do call forwarding")
region = st.selectbox("Region hint", ["Global","United States","United Kingdom","India","Canada","Australia"], index=0)
num_queries = st.number_input("Number of synthetic queries", min_value=3, max_value=200, value=30)
extra_instructions = st.text_area("Extra prompt instructions (optional)", value="Emulate AI Overview style prioritizing common user intents and SERP signals.")

st.subheader("Optional SERP context (paste top titles/snippets / PAA / related searches)")
serp_context = st.text_area("Paste SERP context here (3-10 lines recommended)")

run = st.button("Run AIO-style Fan-Out")

# session storage
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "raw" not in st.session_state:
    st.session_state.raw = ""
if "attempts" not in st.session_state:
    st.session_state.attempts = []

# load optional search console data
sc_lookup = {}
if sc_file is not None:
    try:
        sc_df = pd.read_csv(sc_file)
        if "query" in sc_df.columns:
            sc_df["query_norm"] = sc_df["query"].astype(str).str.strip().str.lower()
            if "clicks" in sc_df.columns:
                total_clicks = sc_df["clicks"].sum() if sc_df["clicks"].sum() > 0 else 1.0
                sc_df["click_norm"] = sc_df["clicks"] / total_clicks
            else:
                sc_df["click_norm"] = 0.0
            sc_lookup = dict(zip(sc_df["query_norm"], sc_df["click_norm"]))
        else:
            st.warning("Search Console CSV missing 'query' column; ignoring SC file.")
            sc_lookup = {}
    except Exception as e:
        st.warning(f"Failed to load SC CSV: {e}")
        sc_lookup = {}

# Run generation
if run:
    st.session_state.df = pd.DataFrame()
    st.session_state.raw = ""
    st.session_state.attempts = []

    if not api_key or not api_key.strip():
        st.error("Gemini API key required.")
    else:
        prompt = build_prompt(seed_query, region, int(num_queries), serp_context or "", extra_instructions)
        st.info("Calling flash models (REST) using provided context...")

        attempts = []
        response_text = None
        used_model = None
        for model_name in models_selected:
            resp = call_model_rest(api_key.strip(), model_name, prompt)
            attempts.append({"model": model_name, "url": resp["url"], "status": resp["status"], "body": (resp["body"] or "")[:800]})
            if resp["ok"]:
                response_text = extract_text_from_response_body(resp["body"])
                if response_text is None:
                    response_text = resp["body"]
                used_model = model_name
                break

        st.session_state.attempts = attempts

        if not response_text:
            st.error("All model attempts failed. See attempt log below.")
        else:
            cleaned = strip_code_fence(response_text)
            parsed = parse_json_array_from_text(cleaned)

            # fallback: maybe it's an array of strings
            if parsed is None:
                try:
                    maybe = json.loads(cleaned)
                    if isinstance(maybe, list) and maybe and isinstance(maybe[0], str):
                        parsed = [{"query": s, "intent":"", "entity":"", "variation_type":"", "region_weight":0.0, "rationale":""} for s in maybe]
                except Exception:
                    parsed = None

            if parsed is None:
                st.warning("Could not parse JSON from model. Showing raw cleaned candidate text for debugging.")
                st.code(cleaned[:4000])
                st.markdown("---")
                st.subheader("Attempt log")
                st.dataframe(pd.DataFrame(attempts[::-1]))
            else:
                # compute serp tokens and lines for heuristics
                serp_tokens = []
                serp_lines = serp_signals_from_context(serp_context)
                for line in serp_lines:
                    serp_tokens.extend(tokenize_for_overlap(line))
                serp_tokens = list(set(serp_tokens))

                rows = []
                for it in parsed:
                    norm = normalize_item(it)
                    q = norm["query"]
                    overlap = compute_serp_overlap_score(q, serp_tokens)
                    sim = compute_best_serp_similarity(q, serp_lines)
                    sc_score = sc_lookup.get(q.strip().lower(), 0.0) if sc_lookup else 0.0
                    rows.append({
                        "query": q,
                        "intent": norm["intent"],
                        "entity": norm["entity"],
                        "variation_type": norm["variation_type"],
                        "region_weight": norm["region_weight"],
                        "rationale": norm["rationale"],
                        "serp_overlap": round(overlap, 3),
                        "serp_similarity": round(sim, 3),
                        "sc_click_norm": round(sc_score, 6)
                    })

                df = pd.DataFrame(rows)
                # sort by combined heuristic: serp_similarity then sc_click then serp_overlap
                df["combined_score"] = (df["serp_similarity"] * 0.6) + (df["sc_click_norm"] * 0.3) + (df["serp_overlap"] * 0.1)
                df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
                st.session_state.df = df
                st.session_state.raw = cleaned

                st.success(f"Generated {len(df)} items with model: {used_model}")
                st.subheader("Generated queries (AIO-style approximation)")
                st.dataframe(df.drop(columns=["combined_score"]), use_container_width=True)

                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="qforia_aio_results.csv", mime="text/csv")
                with col2:
                    st.download_button("Download JSON", data=json.dumps(rows, ensure_ascii=False, indent=2), file_name="qforia_aio_results.json", mime="application/json")

                st.markdown("---")
                st.subheader("Raw cleaned candidate text (truncated)")
                st.code(cleaned[:4000])

# Show attempt log
if st.session_state.attempts:
    st.markdown("---")
    st.subheader("REST attempt log (most recent first)")
    st.dataframe(pd.DataFrame(st.session_state.attempts[::-1]))

st.markdown("""
Notes:
- This app conditions generation on pasted SERP context to approximate AI Overview outputs; it does not have Google internal telemetry.
- Heuristic scores (serp_overlap, serp_similarity, sc_click_norm) are basic approximations and should be used as signals, not ground truth.
- Validate top candidates against Search Console or other telemetry before making large-scale SEO changes.
""")
