# app.py
"""
Qforia - Free-tier (Flash) REST-only Query Fan-Out
REST-only, tries flash models (no google-generativeai client).
Produces a table of synthetic queries with metadata fields.
"""

import streamlit as st
import requests
import json
import re
import pandas as pd
from typing import List, Dict, Any, Optional

# ---------------------------
# Configuration: candidate flash models (no 'models/' prefix)
# ---------------------------
FLASH_MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-flash"
]

BASE_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# ---------------------------
# Helpers
# ---------------------------
def build_prompt(seed: str, region: str, n: int, extra: str = "") -> str:
    """
    Structured prompt asking for JSON array of objects.
    Each object must contain: query, intent, entity, variation_type, region_weight, rationale
    """
    ts = pd.Timestamp.utcnow().isoformat() + "Z"
    prompt = f"""
You are a generator of realistic user search queries for SEO analysis. Output ONLY valid JSON: an array of exactly {n} objects.

Each object must contain the following keys:
- query (string): the user search query text
- intent (string): one of informational, commercial, transactional, navigational, investigational
- entity (string): main entity or brand in the query (or empty string)
- variation_type (string): one of paraphrase, narrowing, expansion, entity-focus, question-form, long-tail, comparative
- region_weight (number): relative weight (0-1) indicating how strongly this variant is region-specific
- rationale (string): one short sentence explaining why this query variant is useful or distinct

Seed query: "{seed}"
Region hint: "{region}"
Timestamp: {ts}

Extra prompt instructions:
{extra}

Return ONLY the JSON array. Do not include explanation or code fences. If the model outputs a code fence, include only the JSON content inside it.
"""
    return prompt.strip()

def call_model_rest(api_key: str, model: str, prompt: str, timeout: int = 40) -> Dict[str, Any]:
    """
    Call the specified flash model. Returns a dict with keys:
    - ok (bool), status, body (text), url (str)
    """
    url = BASE_TEMPLATE.format(model=model)
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        resp = requests.post(url, params={"key": api_key}, json=payload, timeout=timeout)
        return {"ok": resp.status_code == 200, "status": resp.status_code, "body": resp.text, "url": url}
    except Exception as e:
        return {"ok": False, "status": None, "body": str(e), "url": url}

def extract_text_from_response_body(body: str) -> Optional[str]:
    """
    Given the raw JSON body string from the API response, attempt to extract
    the textual candidate. Typical shape:
    {"candidates":[{"content":{"parts":[{"text":"..."}]}}], ...}
    """
    try:
        data = json.loads(body)
    except Exception:
        return None
    # try common path
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
    # fallback: if top-level text present
    for key in ("output", "text", "content"):
        if key in data and isinstance(data[key], str):
            return data[key]
    return None

def strip_code_fence(text: str) -> str:
    """
    Remove triple-backtick fences like ```json ... ``` and surrounding whitespace.
    """
    if not text:
        return text
    # Remove leading/trailing backticks and optional language marker
    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()

def parse_json_array_from_text(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Attempts to parse a JSON array from text. Handles code fences and extra text.
    """
    if text is None:
        return None
    t = strip_code_fence(text)
    # direct parse
    try:
        parsed = json.loads(t)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    # search for first JSON array occurrence
    m = re.search(r"\[[\s\S]*\]", t)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return None
    return None

def normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure fields exist and coerce types.
    """
    return {
        "query": item.get("query","") if isinstance(item, dict) else str(item),
        "intent": item.get("intent","") if isinstance(item, dict) else "",
        "entity": item.get("entity","") if isinstance(item, dict) else "",
        "variation_type": item.get("variation_type","") if isinstance(item, dict) else "",
        "region_weight": float(item.get("region_weight", 0)) if isinstance(item, dict) and item.get("region_weight") is not None else 0.0,
        "rationale": item.get("rationale","") if isinstance(item, dict) else ""
    }

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(layout="wide", page_title="Qforia-free (Flash) — Query Fan-Out")
st.title("Qforia-free — Query Fan-Out (Flash model, REST only)")

with st.sidebar:
    st.header("Authentication")
    st.write("Paste your Gemini API key (AI Studio / free-tier).")
    api_key = st.text_input("Gemini API Key", type="password")
    st.markdown("---")
    st.write("Model selection (flash models only). The app will try models in order until a response succeeds.")
    # Show candidates and allow override
    models_selected = st.multiselect("Flash model candidates (order matters)", FLASH_MODEL_CANDIDATES, default=FLASH_MODEL_CANDIDATES[:2])
    if not models_selected:
        models_selected = FLASH_MODEL_CANDIDATES[:2]

st.subheader("Fan-out configuration")
seed_query = st.text_input("Seed query", value="how to do call forwarding")
region = st.selectbox("Region hint", ["Global","United States","United Kingdom","India","Canada","Australia"], index=0)
num_queries = st.number_input("Number of synthetic queries", min_value=3, max_value=200, value=30)
extra_instructions = st.text_area("Extra prompt instructions (optional)", value="")
run_button = st.button("Run Fan-Out (flash models)")

# session state for results
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "raw" not in st.session_state:
    st.session_state.raw = ""
if "attempts" not in st.session_state:
    st.session_state.attempts = []

# Run
if run_button:
    st.session_state.df = pd.DataFrame()
    st.session_state.raw = ""
    st.session_state.attempts = []

    if not api_key or not api_key.strip():
        st.error("Gemini API key is required.")
    else:
        prompt = build_prompt(seed_query, region, int(num_queries), extra_instructions)
        st.info("Attempting flash models in order...")

        success = False
        attempts_log = []
        extracted_text = None
        used_model = None

        for model_name in models_selected:
            resp = call_model_rest(api_key.strip(), model_name, prompt)
            attempts_log.append({"url": resp["url"], "status": resp["status"], "body": (resp["body"] or "")[:800], "model": model_name})
            if resp["ok"]:
                extracted_text = extract_text_from_response_body(resp["body"])
                used_model = model_name
                # Some responses include backticks/code fence; strip before parsing
                if extracted_text is None:
                    # fallback: attempt to parse top-level body text
                    extracted_text = resp["body"]
                success = True
                break

        st.session_state.attempts = attempts_log

        if not success:
            st.error("Flash-model REST call(s) failed. See attempt log below.")
        else:
            # clean and parse
            cleaned = strip_code_fence(extracted_text)
            parsed_array = parse_json_array_from_text(cleaned)

            if parsed_array is None:
                # try alternative: if the response is a JSON array of strings (older shape), convert into objects
                try:
                    arr = json.loads(cleaned)
                    if isinstance(arr, list) and arr and isinstance(arr[0], str):
                        # convert each string to an object with defaults
                        parsed_array = [{"query": s, "intent": "", "entity": "", "variation_type": "", "region_weight": 0.0, "rationale": ""} for s in arr]
                except Exception:
                    parsed_array = None

            if parsed_array is None:
                st.warning("Could not parse structured JSON from model output. Showing raw candidate text for debugging.")
                st.code(cleaned[:4000])
                st.markdown("---")
                st.subheader("REST attempts (latest shown first)")
                st.dataframe(pd.DataFrame(attempts_log[::-1]))
            else:
                # normalize items into table
                rows = [normalize_item(it) for it in parsed_array]
                df = pd.DataFrame(rows)
                st.session_state.df = df
                st.session_state.raw = cleaned

                st.success(f"Generated {len(df)} synthetic queries using model: {used_model}")
                st.subheader("Synthetic queries (table)")
                st.dataframe(df, use_container_width=True)

                # downloads
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", data=csv_bytes, file_name="qforia_flash_results.csv", mime="text/csv")
                st.download_button("Download JSON", data=json.dumps(rows, ensure_ascii=False, indent=2), file_name="qforia_flash_results.json", mime="application/json")

                # small preview of raw
                st.markdown("---")
                st.subheader("Raw model candidate text (cleaned, truncated)")
                st.code(cleaned[:4000])

# show attempts log when present
if st.session_state.attempts:
    st.markdown("---")
    st.subheader("REST attempt log (most recent first)")
    st.dataframe(pd.DataFrame(st.session_state.attempts[::-1]))

st.markdown("""
Notes:
- This app uses free-tier flash models only. If all attempts return 404/403, your API key/project does not have flash access.
- Flash models have lower fidelity than pro models.
- If the model returns its JSON inside backticks or extra text, the app tries to strip and parse it.
""")
