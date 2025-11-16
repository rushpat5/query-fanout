# app.py
"""
Qforia-region Streamlit app with Google Gemini (Vertex AI) example.
- Uses the google-generativeai client if available.
- Replace PROJECT_ID / LOCATION / MODEL_NAME with your values.
- Set GOOGLE_API_KEY OR use ADC via GOOGLE_APPLICATION_CREDENTIALS to a service account JSON.
Notes: confirm the client API and model names against Google Cloud docs for your account/region.
"""

import os
import re
import json
import time
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np

# Attempt to import google.generativeai; if unavailable, instruct the user.
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

# -------------------- CONFIG (replace placeholders) --------------------
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-gcp-project-id")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")   # e.g., us-central1, europe-west4
MODEL_NAME = os.getenv("GEMINI_MODEL", "models/text-bison-001")  # replace with your model resource name if needed
# You can set GOOGLE_API_KEY or use GOOGLE_APPLICATION_CREDENTIALS (ADC) for authentication.
# ----------------------------------------------------------------------

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

# -------------------- Helpers --------------------
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

def extract_json_array(text: str):
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

# -------------------- Gemini (Vertex AI) calls --------------------
def configure_genai(api_key: Optional[str] = None):
    """Configure google.generativeai client (if available). Accepts API key or relies on ADC."""
    if not GENAI_AVAILABLE:
        raise RuntimeError("google.generativeai package not installed. pip install google-generativeai")
    # If you have an explicit API key:
    explicit_key = api_key or os.getenv("GOOGLE_API_KEY")
    if explicit_key:
        genai.configure(api_key=explicit_key)
    else:
        # Rely on ADC (Application Default Credentials)
        # No explicit configuration required if ADC is set via GOOGLE_APPLICATION_CREDENTIALS
        # But set project & location for convenience
        genai.configure(project=PROJECT_ID, location=LOCATION)

def call_gemini_text(prompt: str, model: str = None, temperature: float = 0.0, max_output_tokens: int = 512) -> str:
    """
    Call a text generation endpoint on Vertex AI (Gemini / Bison family) via google.generativeai.
    Returns the raw text produced by the model.
    NOTE: Validate model identifier and SDK usage against your installed google-generativeai version.
    """
    if not GENAI_AVAILABLE:
        raise RuntimeError("google.generativeai not available. Install google-generativeai and configure credentials.")
    model_to_use = model or MODEL_NAME
    # Example using genai.client
    # The actual call shape depends on SDK version; this is a commonly used pattern:
    try:
        # Use genai.generate_text or genai.chat.create depending on client version
        if hasattr(genai, "generate_text"):
            resp = genai.generate_text(model=model_to_use, prompt=prompt, max_output_tokens=max_output_tokens, temperature=temperature)
            # Many client versions store text at resp.text or resp["candidates"][0]["output"]
            if isinstance(resp, dict) and "candidates" in resp and resp["candidates"]:
                return resp["candidates"][0].get("output", "")
            # fallback
            return getattr(resp, "text", "") or ""
        elif hasattr(genai, "chat"):
            # genai.chat.create is another common interface
            chat_resp = genai.chat.create(model=model_to_use, messages=[{"role":"user","content":prompt}], temperature=temperature, max_output_tokens=max_output_tokens)
            # find content
            # different SDKs provide different shapes
            if hasattr(chat_resp, "candidates") and chat_resp.candidates:
                return chat_resp.candidates[0].content
            # try messages
            if hasattr(chat_resp, "output"):
                return chat_resp.output
            return str(chat_resp)
        else:
            # Last resort: try a generic method
            resp = genai.create(model=model_to_use, prompt=prompt)
            return str(resp)
    except Exception as e:
        # surface the error for debugging in the UI
        raise

# Embeddings via Vertex AI: some SDKs expose embeddings via genai.get_embeddings / genai.embeddings
def gemini_embeddings(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    """Compute embeddings via the same client. Implementation depends on sdk version."""
    if not GENAI_AVAILABLE:
        raise RuntimeError("google.generativeai not installed.")
    emb_model = model or "embedding-gecko-001"  # placeholder; pick available embedding model in your project
    if hasattr(genai, "embeddings") and hasattr(genai.embeddings, "create"):
        resp = genai.embeddings.create(model=emb_model, input=texts)
        # resp.data[*].embedding typical
        out = []
        for item in resp.data:
            out.append(item.embedding)
        return out
    elif hasattr(genai, "get_embeddings"):
        resp = genai.get_embeddings(model=emb_model, input=texts)
        return resp["embeddings"]
    else:
        raise RuntimeError("Your google.generativeai client version does not expose embeddings in an expected way; check docs.")

# -------------------- Utilities --------------------
def cosine_sim(a: List[float], b: List[float]) -> float:
    import numpy as _np
    a = _np.array(a, dtype=float)
    b = _np.array(b, dtype=float)
    if _np.linalg.norm(a) == 0 or _np.linalg.norm(b) == 0:
        return 0.0
    return float(_np.dot(a, b) / (_np.linalg.norm(a) * _np.linalg.norm(b)))

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Qforia-region (Gemini)", layout="wide")
st.title("Qforia-region — Query Fan-Out Simulator (Gemini)")

with st.sidebar:
    st.header("Configuration")
    st.write("Authentication: Set GOOGLE_APPLICATION_CREDENTIALS env var to a service-account JSON, or set GOOGLE_API_KEY.")
    api_key_box = st.text_input("Google API Key (optional)", type="password")
    project_box = st.text_input("Project ID", value=PROJECT_ID)
    location_box = st.text_input("Location", value=LOCATION)
    model_box = st.text_input("Model resource name", value=MODEL_NAME)
    num_queries = st.number_input("Number of synthetic queries", min_value=5, max_value=200, value=DEFAULT_NUM_QUERIES)
    surface = st.radio("Search Mode", AVAILABLE_SURFACES)

    st.markdown("---")
    st.subheader("Region / Locale")
    region_choice = st.selectbox("Choose a region", [r["name"] for r in REGIONS], index=0)
    region = next((r for r in REGIONS if r["name"] == region_choice), REGIONS[0])
    custom_language = st.text_input("Override language (optional)", value=region.get("language"))
    if custom_language:
        region = region.copy()
        region["language"] = custom_language

    st.markdown("---")
    st.write("Prompt tuning")
    extra_instructions = st.text_area("Extra prompt instructions (optional)", value="")

col1, col2 = st.columns([2,1])
with col1:
    seed_query = st.text_input("Seed query", value="how to do call forwarding")
    run_btn = st.button("Run Fan-Out 🚀")
with col2:
    st.write("Export")
    export_csv = st.button("Export last results to CSV")
    export_json = st.button("Export last results to JSON")

if "last_df" not in st.session_state:
    st.session_state["last_df"] = pd.DataFrame()
if "embeddings" not in st.session_state:
    st.session_state["embeddings"] = []

if run_btn and seed_query.strip():
    # configure client
    try:
        configure_genai(api_key_box if api_key_box else None)
        # update globals from UI
        global PROJECT_ID, LOCATION, MODEL_NAME
        PROJECT_ID = project_box or PROJECT_ID
        LOCATION = location_box or LOCATION
        MODEL_NAME = model_box or MODEL_NAME

        prompt = make_prompt(seed_query, surface, region, int(num_queries), extra_instructions)
        with st.spinner("Calling Gemini..."):
            raw = call_gemini_text(prompt, model=MODEL_NAME, temperature=0.0, max_output_tokens=1024)
    except Exception as e:
        st.error(f"Gemini call/config error: {e}")
        raw = None

    parsed = None
    if raw:
        parsed = extract_json_array(raw)
        if parsed is None:
            st.error("Could not parse JSON from Gemini output. Showing raw response for debugging.")
            st.code(raw)
        else:
            rows = []
            for i, item in enumerate(parsed):
                rows.append({
                    "rank": i+1,
                    "query": item.get("query",""),
                    "intent": item.get("intent","informational"),
                    "entity": item.get("entity",""),
                    "variation_type": item.get("variation_type",""),
                    "rationale": item.get("rationale","")
                })
            df = pd.DataFrame(rows)
            st.session_state["last_df"] = df

            # embeddings (best-effort)
            try:
                texts = df["query"].tolist()
                emb = gemini_embeddings(texts)
                st.session_state["embeddings"] = emb
                st.success(f"Generated {len(df)} queries and computed embeddings (Gemini).")
            except Exception as e:
                st.warning(f"Query generation succeeded but embeddings failed or not available via SDK: {e}")
                st.session_state["embeddings"] = []

if not st.session_state["last_df"].empty:
    df = st.session_state["last_df"]
    st.subheader("Synthetic Queries")
    st.dataframe(df)

    st.markdown("---")
    st.subheader("Filters")
    intents = df["intent"].unique().tolist()
    sel_intents = st.multiselect("Filter by intent", options=intents, default=intents)
    filtered = df[df["intent"].isin(sel_intents)]
    st.write(f"Showing {len(filtered)} queries after filtering.")
    st.dataframe(filtered)

    if export_csv:
        st.download_button("Download CSV", data=filtered.to_csv(index=False).encode("utf-8"), file_name="qforia_gen_region.csv", mime="text/csv")
    if export_json:
        st.download_button("Download JSON", data=filtered.to_json(orient="records", force_ascii=False), file_name="qforia_gen_region.json", mime="application/json")

    st.markdown("---")
    st.subheader("Passage → Query (semantic matching)")
    passage = st.text_area("Paste page passage/paragraph (optional)")
    match_btn = st.button("Match passage to generated queries")

    if match_btn and passage.strip():
        if not st.session_state["embeddings"]:
            st.error("No embeddings available; re-run generation and ensure embedding-capable model and client are configured.")
        else:
            try:
                pass_emb = gemini_embeddings([passage])[0]
                scores = []
                for i, q_emb in enumerate(st.session_state["embeddings"]):
                    sim = cosine_sim(pass_emb, q_emb)
                    scores.append((df.iloc[i]["query"], sim, df.iloc[i]["intent"], df.iloc[i]["rationale"]))
                scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
                out = [{"query": s[0], "score": round(s[1],4), "intent": s[2], "rationale": s[3]} for s in scores_sorted[:20]]
                st.table(pd.DataFrame(out))
            except Exception as e:
                st.error(f"Embeddings matching failed: {e}")

st.markdown("---")
st.caption("Notes: This app uses google-generativeai if available. Confirm model names, quota, and pricing in Google Cloud Console. Region hints bias outputs but do not expose or replicate Google internal fan-out lists.")