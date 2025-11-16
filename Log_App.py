# app.py
"""
Qforia-region (Gemini REST) — model-probing edition.

Behavior:
 - Provide a Gemini / Generative API key in the UI.
 - Optionally enter a model resource name. If left blank, the app probes common model candidates
   across /v1 and /v1beta2 endpoints until one responds.
 - On success, uses that model for generation.
 - Shows raw response when JSON parsing fails to aid debugging.

Install: pip install streamlit pandas requests
Run: streamlit run app.py
"""

import os, json, re, time
from typing import List, Dict, Any, Optional
import difflib
import requests
import streamlit as st
import pandas as pd

# ---------------- Config (edit if you want different defaults) ----------------
DEFAULT_NUM_QUERIES = 30
AVAILABLE_SURFACES = ["AI Overview", "AI Mode"]

# Candidate model names to probe when user doesn't supply one.
# Expand this list with models you expect in your account.
MODEL_CANDIDATES = [
    "text-bison-001",
    "models/text-bison-001",
    "chat-bison-001",
    "models/chat-bison-001",
    "gemini-1.5-pro",
    "models/gemini-1.5-pro",
    "gemini-pro",
    "models/gemini-pro",
    "gemini-1.0",
    "models/gemini-1.0",
]

# API base path candidates (try newer v1 first, then v1beta2)
API_BASES = [
    "https://generativelanguage.googleapis.com/v1/models",
    "https://generativelanguage.googleapis.com/v1beta2/models",
]

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

def similarity_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a_low = a.lower()
    b_low = b.lower()
    if a_low in b_low or b_low in a_low:
        return 1.0
    return difflib.SequenceMatcher(None, a_low, b_low).ratio()

# ---------------- REST call with explicit model & base ----------------
def call_generative_rest_with_base(prompt: str, api_key: str, base: str, model: str, temperature: float = 0.0, max_output_tokens: int = 512) -> Dict[str,Any]:
    """
    Returns tuple (success_bool, response_text_or_json, http_status, http_text)
    """
    url = f"{base.rstrip('/')}/{model}:generateText"
    params = {"key": api_key}
    payload = {
        "prompt": {"text": prompt},
        "temperature": float(temperature),
        "maxOutputTokens": int(max_output_tokens),
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, params=params, json=payload, headers=headers, timeout=60)
    status = resp.status_code
    body_text = resp.text
    if status != 200:
        return {"ok": False, "status": status, "body": body_text}
    try:
        data = resp.json()
    except Exception:
        return {"ok": True, "status": status, "body": body_text}
    # try to extract string candidate from common shapes
    # 1) data["candidates"][0]["output"]
    if isinstance(data, dict):
        if "candidates" in data and isinstance(data["candidates"], list) and data["candidates"]:
            cand = data["candidates"][0]
            for k in ("output","content","text","displayText"):
                if isinstance(cand, dict) and k in cand and isinstance(cand[k], str):
                    return {"ok": True, "status": status, "body": cand[k], "raw": data}
        for k in ("output","response","result","content"):
            if k in data and isinstance(data[k], str):
                return {"ok": True, "status": status, "body": data[k], "raw": data}
    # fallback: return entire JSON
    return {"ok": True, "status": status, "body": json.dumps(data, indent=2), "raw": data}

def find_working_model(prompt: str, api_key: str, explicit_model: Optional[str]=None, candidates: Optional[List[str]] = None) -> Dict[str,Any]:
    """
    Try to discover a model and base that work. Returns dict with keys:
      success (bool), base, model, response (string), attempts (list)
    """
    attempts = []
    if explicit_model:
        # try explicit_model across API_BASES
        for base in API_BASES:
            r = call_generative_rest_with_base(prompt, api_key, base, explicit_model)
            attempts.append({"base": base, "model": explicit_model, "result": r})
            if r.get("ok"):
                return {"success": True, "base": base, "model": explicit_model, "response": r.get("body"), "attempts": attempts}
        return {"success": False, "attempts": attempts}
    # no explicit model: try candidates across bases
    candidates = candidates or MODEL_CANDIDATES
    for base in API_BASES:
        for model in candidates:
            r = call_generative_rest_with_base(prompt, api_key, base, model)
            attempts.append({"base": base, "model": model, "result": r})
            if r.get("ok"):
                return {"success": True, "base": base, "model": model, "response": r.get("body"), "attempts": attempts}
    return {"success": False, "attempts": attempts}

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Qforia-region (Gemini REST probe)", layout="wide")
st.title("Qforia-region — Query Fan-Out Simulator (Gemini REST probe)")

with st.sidebar:
    st.header("Configuration")
    st.write("Provide a Gemini / Generative API Key (preferred). Optional: enter the exact model resource name if you know it.")
    api_key = st.text_input("Gemini API Key (optional)", type="password")
    user_model = st.text_input("Optional model resource name (leave blank to auto-probe)")
    st.caption("If probing fails, paste the first failed attempts shown by the app and I'll help adjust the model name.")

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
if "last_raw" not in st.session_state:
    st.session_state["last_raw"] = ""

if run_btn and seed_query.strip():
    prompt = make_prompt(seed_query, surface, region, int(num_queries), extra_instructions)
    if not api_key:
        st.error("No API key provided. Enter a Gemini API key to use REST path (preferred).")
    else:
        st.info("Probing for a reachable model/resource. This may take a few seconds.")
        probe = find_working_model(prompt, api_key, explicit_model=user_model.strip() if user_model.strip() else None)
        if not probe["success"]:
            st.error("No usable model found with REST probing. See attempts below.")
            attempts = probe["attempts"]
            # show summary of attempts and HTTP codes for diagnosis
            rows = []
            for a in attempts:
                r = a["result"]
                rows.append({
                    "base": a["base"],
                    "model": a["model"],
                    "ok": r.get("ok", False),
                    "status": r.get("status"),
                    "short": (r.get("body") or "")[:200].replace("\n"," ")
                })
            st.dataframe(pd.DataFrame(rows))
            st.error("Common causes: model name not available to your key, wrong model identifier, or account not enabled for Generative API. Provide a valid model resource name or verify your API key & project permissions.")
        else:
            chosen_base = probe["base"]
            chosen_model = probe["model"]
            raw_body = probe["response"]
            st.success(f"Using model {chosen_model} at base {chosen_base}")
            st.session_state["last_raw"] = raw_body or ""
            parsed = extract_json_array(raw_body or "")
            if parsed is None:
                st.warning("Model returned text that could not be parsed as JSON. Showing raw output for debugging (truncated):")
                st.code((raw_body or "")[:4000])
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

# Display results
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

# Show last raw response if present
if st.session_state.get("last_raw"):
    st.markdown("---")
    st.subheader("Last raw response (truncated)")
    st.code((st.session_state["last_raw"] or "")[:4000])

st.markdown("---")
st.caption("If probing fails, paste the attempts table into your next message and I will adapt candidate model names or help verify permissions. This app probes common model names and both v1/v1beta2 endpoints; model availability depends on your Google account & key.")
