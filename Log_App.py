# app.py
"""
Qforia-region (Gemini) - REST-first implementation that accepts a Gemini/Generative API key.
- Enter a Gemini API key in the UI. If provided, the app calls the public Generative Language REST endpoint.
- If no API key is provided, the app attempts to use the installed google.generativeai client as a fallback.
- Passage→query matching uses substring + difflib similarity fallback (no embeddings required).
Notes:
 - Install: pip install streamlit pandas requests
 - Run: streamlit run app.py
 - Change MODEL_NAME to a model your key can access if necessary.
"""

import os
import json
import re
import time
from typing import List, Dict, Any, Optional
import difflib

import requests
import streamlit as st
import pandas as pd
import numpy as np

# Try optional google.generativeai import as a fallback
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

# ---------------- Config (change MODEL_NAME if your account requires)
MODEL_NAME = os.getenv("GEMINI_MODEL", "text-bison-001")  # typical public model name; change if needed
GENERATIVE_REST_BASE = "https://generativelanguage.googleapis.com/v1beta2/models"
DEFAULT_NUM_QUERIES = 30
AVAILABLE_SURFACES = ["AI Overview", "AI Mode"]

REGIONS = [
    {"name": "United States", "code": "US", "language": "en-US"},
    {"name": "United Kingdom", "code": "GB", "language": "en-GB"},
    {"name": "India", "code": "IN", "language": "en-IN"},
    {"name": "Canada", "code": "CA", "language": "en-CA"},
    {"name": "Australia", "code": "AU", "language": "en-AU"},
    {"name": "Germany", "code": "DE", "language": "de-DE"},
    {"name": "France", "code": "FR", "language": "fr-FR"},
]

# ---------------- Helpers ----------------
def make_prompt(seed_query: str, surface: str, region: Dict[str,str], num_queries: int, extra_instructions: str="") -> str:
    ts = pd.Timestamp.utcnow().isoformat() + "Z"
    region_hint = f"{region.get('name')} (country_code={region.get('code')}, language={region.get('language')})"
    prompt = f"""
You are a utility assistant that MUST OUTPUT valid JSON only: a JSON array of exactly {num_queries} objects.

Each object must have these fields:
 - query (string)
 - intent (string): informational, commercial, transactional, navigational, investigational
 - entity (string)
 - variation_type (string): paraphrase, narrowing, expansion, entity-focus, question-form, long-tail, comparative
 - rationale (string): 1-2 sentence justification

Seed query: "{seed_query}"
Target surface: {surface}
Region hint: {region_hint}
Timestamp: {ts}

Requirements:
 - Output EXACTLY the JSON array and nothing else (no explanation, no fences).
 - Bias wording and entities to the locale.
 - Keep queries concise (4-12 words typical), except when explicitly long-tail.
 - Diversify intents and variation types.

Extra instructions:
{extra_instructions}

Produce the JSON now.
"""
    return prompt.strip()

def extract_json_array(text: str) -> Optional[List[Dict[str,Any]]]:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None

# ---------------- REST-based call (API key path) ----------------
def call_generative_rest(prompt: str, api_key: str, model: str = MODEL_NAME, temperature: float = 0.0, max_output_tokens: int = 512) -> str:
    """
    Call the Generative Language REST endpoint using an API key.
    Endpoint: https://generativelanguage.googleapis.com/v1beta2/models/{model}:generateText?key=API_KEY
    Body shape: {"prompt": {"text": prompt}, "temperature": ..., "maxOutputTokens": ...}
    Response parsing is best-effort: many versions place text in 'candidates'[0]['output'] or similar.
    """
    url = f"{GENERATIVE_REST_BASE}/{model}:generateText"
    params = {"key": api_key}
    payload = {
        "prompt": {"text": prompt},
        "temperature": temperature,
        "maxOutputTokens": int(max_output_tokens),
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, params=params, json=payload, headers=headers, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Generative REST error {resp.status_code}: {resp.text}")
    data = resp.json()
    # Try common shapes
    # 1) data.get('candidates')[0].get('output')
    if isinstance(data, dict):
        if "candidates" in data and isinstance(data["candidates"], list) and data["candidates"]:
            cand = data["candidates"][0]
            # candidate might have 'output' or 'content' or 'text'
            for k in ("output","content","text","displayText"):
                if isinstance(cand, dict) and k in cand and isinstance(cand[k], str):
                    return cand[k]
        # 2) data.get('output') or data.get('response')
        for k in ("output","response","result"):
            if k in data and isinstance(data[k], str):
                return data[k]
    # fallback: return whole body as text for debugging
    return json.dumps(data, indent=2)

# ---------------- Fallback SDK call (if no API key) ----------------
def call_genai_sdk(prompt: str, model: str = MODEL_NAME, temperature: float = 0.0, max_output_tokens: int = 512) -> str:
    """
    Best-effort usage of installed google.generativeai client.
    This may need adaptation per client version. The app will show raw output if parsing fails.
    """
    if not GENAI_AVAILABLE:
        raise RuntimeError("google.generativeai client not installed.")
    # try chat.create
    try:
        if hasattr(genai, "chat") and hasattr(genai.chat, "create"):
            r = genai.chat.create(model=model, messages=[{"role":"user","content":prompt}], temperature=temperature, max_output_tokens=max_output_tokens)
            # try common attributes
            if hasattr(r, "candidates") and r.candidates:
                c = r.candidates[0]
                return getattr(c, "content", getattr(c, "message", str(c)))
            if hasattr(r, "output"):
                return getattr(r, "output")
            return str(r)
    except Exception:
        pass
    # try generate_text
    try:
        if hasattr(genai, "generate_text"):
            r = genai.generate_text(model=model, prompt=prompt, temperature=temperature, max_output_tokens=max_output_tokens)
            if isinstance(r, dict) and "candidates" in r and r["candidates"]:
                return r["candidates"][0].get("output") or r["candidates"][0].get("content","")
            if hasattr(r, "text"):
                return getattr(r, "text")
            return str(r)
    except Exception:
        pass
    # last fallback: stringified object
    raise RuntimeError("Installed google.generativeai client does not expose a supported method in this code path.")

# ---------------- Simple passage->query similarity (no embeddings) ----------------
def similarity_score(a: str, b: str) -> float:
    """Combine substring check and difflib ratio for robust matching (0..1)."""
    if not a or not b:
        return 0.0
    a_low = a.lower()
    b_low = b.lower()
    if a_low in b_low or b_low in a_low:
        return 1.0
    # difflib ratio
    return difflib.SequenceMatcher(None, a_low, b_low).ratio()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Qforia-region (Gemini REST)", layout="wide")
st.title("Qforia-region — Query Fan-Out Simulator (Gemini REST)")

with st.sidebar:
    st.header("Configuration")
    st.write("Provide a Gemini / Generative API Key (or set GOOGLE_API_KEY env var).")
    api_key = st.text_input("Gemini API Key (optional)", type="password")
    st.markdown("---")
    st.caption("The app will use the API key path (REST) when you enter a key; otherwise it attempts the installed SDK. Use a key for the simplest flow.")

col1, col2 = st.columns([3,1])
with col1:
    seed_query = st.text_input("Seed query", value="how to do call forwarding")
    region_choice = st.selectbox("Region", [r["name"] for r in REGIONS], index=0)
    region = next((r for r in REGIONS if r["name"] == region_choice), REGIONS[0])
    num_queries = st.number_input("Number of synthetic queries", min_value=5, max_value=200, value=DEFAULT_NUM_QUERIES)
    surface = st.selectbox("Target surface", AVAILABLE_SURFACES)
    extra_instructions = st.text_area("Extra prompt instructions (optional)", value="")
    run_btn = st.button("Run Fan-Out 🚀")
with col2:
    st.write("Export / Actions")
    export_csv = st.button("Export last results to CSV")
    export_json = st.button("Export last results to JSON")

if "last_df" not in st.session_state:
    st.session_state["last_df"] = pd.DataFrame()
if "raw_last" not in st.session_state:
    st.session_state["raw_last"] = ""

if run_btn and seed_query.strip():
    prompt = make_prompt(seed_query, surface, region, int(num_queries), extra_instructions)
    raw_output = None
    # Prefer API key + REST if api_key provided
    if api_key:
        try:
            raw_output = call_generative_rest(prompt, api_key=api_key, model=MODEL_NAME, temperature=0.0, max_output_tokens=1024)
        except Exception as e:
            st.error(f"Generative REST call failed: {e}")
            raw_output = None
    else:
        # Try SDK fallback
        if GENAI_AVAILABLE:
            try:
                # Configure SDK to use ADC if needed; genai.configure(api_key=...) could be used if key available
                raw_output = call_genai_sdk(prompt, model=MODEL_NAME, temperature=0.0, max_output_tokens=1024)
            except Exception as e:
                st.error(f"SDK call failed: {e}")
                raw_output = None
        else:
            st.error("No API key provided and google.generativeai is not installed. Provide a Gemini API key to use REST path.")
            raw_output = None

    st.session_state["raw_last"] = raw_output or ""

    parsed = None
    if raw_output:
        parsed = extract_json_array(raw_output)
        if parsed is None:
            st.error("Could not parse JSON from model output. Showing raw output (truncated):")
            st.code((raw_output or "")[:4000])
        else:
            rows = []
            for i, item in enumerate(parsed):
                rows.append({
                    "rank": i+1,
                    "query": item.get("query","").strip(),
                    "intent": item.get("intent","informational"),
                    "entity": item.get("entity",""),
                    "variation_type": item.get("variation_type",""),
                    "rationale": item.get("rationale","")
                })
            df = pd.DataFrame(rows)
            st.session_state["last_df"] = df

# Display results if present
if not st.session_state["last_df"].empty:
    df = st.session_state["last_df"]
    st.subheader("Synthetic Queries")
    st.dataframe(df)

    if export_csv:
        st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="qforia_region_export.csv", mime="text/csv")
    if export_json:
        st.download_button("Download JSON", data=df.to_json(orient="records", force_ascii=False), file_name="qforia_region_export.json", mime="application/json")

    st.markdown("---")
    st.subheader("Passage → Query mapping (no embeddings)")
    passage = st.text_area("Paste page passage/paragraph (optional)")
    match_btn = st.button("Find top matching generated queries")
    if match_btn and passage.strip():
        scores = []
        for idx, row in df.iterrows():
            score = similarity_score(row["query"], passage)
            scores.append((row["query"], round(score,4), row["intent"], row["rationale"]))
        scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
        st.table(pd.DataFrame([{"query": s[0], "score": s[1], "intent": s[2], "rationale": s[3]} for s in scores_sorted[:20]]))

st.markdown("---")
st.caption("This app uses a REST-first approach when you provide a Gemini API key. If you prefer full SDK ADC/service-account flows or embeddings, I can add those next. Region hints bias outputs but do not guarantee alignment with live query logs.")
