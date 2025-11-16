# app.py
"""
Qforia-region (Gemini REST) — deterministic API usage.

Usage:
 - Option A (API key): supply Gemini API key in the UI and enter exact model resource name (e.g., models/text-bison-001).
 - Option B (ADC/service account): leave API key blank, set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON with appropriate permissions,
   then supply model resource name in the UI.

Install:
 pip install streamlit pandas requests google-auth

Run:
 streamlit run app.py
"""

import os
import json
import re
from typing import Optional, List, Dict, Any

import requests
import streamlit as st
import pandas as pd

# Optional google auth for ADC bearer
try:
    import google.auth
    import google.auth.transport.requests
    ADC_AVAILABLE = True
except Exception:
    ADC_AVAILABLE = False

# Config
GENERATIVE_BASE = "https://generativelanguage.googleapis.com"
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

# REST call using API key
def call_with_api_key(prompt: str, api_key: str, model_resource: str, max_tokens: int = 1024, temperature: float = 0.0) -> Dict[str,Any]:
    # model_resource should be like "models/text-bison-001" or "models/gemini-1.5-pro"
    url = f"{GENERATIVE_BASE}/v1/{model_resource}:generateText"
    params = {"key": api_key}
    payload = {"prompt": {"text": prompt}, "maxOutputTokens": int(max_tokens), "temperature": float(temperature)}
    headers = {"Content-Type": "application/json"}
    r = requests.post(url, params=params, json=payload, headers=headers, timeout=60)
    return {"status": r.status_code, "text": r.text, "ok": r.status_code == 200}

# REST call using ADC bearer token
def call_with_adc(prompt: str, model_resource: str, max_tokens: int = 1024, temperature: float = 0.0) -> Dict[str,Any]:
    if not ADC_AVAILABLE:
        return {"status": None, "text": "ADC not available (google-auth missing)", "ok": False}
    # Acquire default credentials and get access token
    creds, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    token = creds.token
    if not token:
        return {"status": None, "text": "Failed to obtain ADC access token", "ok": False}
    url = f"{GENERATIVE_BASE}/v1/{model_resource}:generateText"
    payload = {"prompt": {"text": prompt}, "maxOutputTokens": int(max_tokens), "temperature": float(temperature)}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    return {"status": r.status_code, "text": r.text, "ok": r.status_code == 200}

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Qforia-region (Gemini REST exact model)", layout="wide")
st.title("Qforia-region — Query Fan-Out Simulator (Gemini REST)")

with st.sidebar:
    st.header("Authentication / Model")
    st.write("Preferred: supply API key *and* the exact model resource name (e.g., models/text-bison-001).")
    st.write("Alternative: leave API key blank and set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON; then supply model resource name.")
    api_key = st.text_input("Gemini API Key (optional)", type="password")
    model_resource = st.text_input("Model resource (required) — e.g., models/text-bison-001")
    st.caption("You must supply the exact model resource name that your key/account can access. Use Console > Generative AI > Models to find it.")

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
    st.write("Export")
    export_csv = st.button("Export last results to CSV")
    export_json = st.button("Export last results to JSON")

if "last_df" not in st.session_state:
    st.session_state["last_df"] = pd.DataFrame()
if "last_raw" not in st.session_state:
    st.session_state["last_raw"] = ""

if run_btn:
    if not model_resource or not model_resource.strip():
        st.error("Model resource is required. Please open Google Cloud Console > Generative AI > Models and copy the model resource name (e.g., models/text-bison-001).")
    else:
        prompt = make_prompt(seed_query, surface, region, int(num_queries), extra_instructions)
        # Prefer API key path when API key provided
        if api_key and api_key.strip():
            st.info("Calling Generative REST with API key...")
            result = call_with_api_key(prompt, api_key.strip(), model_resource.strip())
        else:
            st.info("No API key provided; attempting ADC/service-account bearer token path...")
            result = call_with_adc(prompt, model_resource.strip())

        st.session_state["last_raw"] = result.get("text","")
        if not result.get("ok"):
            st.error(f"Generative REST error {result.get('status')}: {result.get('text')}")
            st.code((result.get("text") or "")[:4000])
        else:
            # success
            raw = result.get("text") or ""
            parsed = extract_json_array(raw)
            if parsed is None:
                st.warning("Model returned text that could not be parsed as JSON. Showing raw output for debugging (truncated):")
                st.code(raw[:4000])
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
                st.session_state["last_df"] = pd.DataFrame(rows)
                st.success("Generated synthetic queries.")

# Display results
if not st.session_state["last_df"].empty:
    df = st.session_state["last_df"]
    st.subheader("Synthetic Queries")
    st.dataframe(df)
    if export_csv:
        st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="qforia_region_export.csv", mime="text/csv")
    if export_json:
        st.download_button("Download JSON", data=df.to_json(orient="records", force_ascii=False), file_name="qforia_region_export.json", mime="application/json")

if st.session_state.get("last_raw"):
    st.markdown("---")
    st.subheader("Last raw response (truncated)")
    st.code((st.session_state.get("last_raw") or "")[:4000])

st.caption("If you still get 404: verify the model resource name in Cloud Console and that your API key/account has Generative API access. If you get a 403: check permissions or use a service account with proper roles.")
