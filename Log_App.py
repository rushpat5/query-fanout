# app.py
"""
Qforia-region Streamlit app (Gemini) — simplified UI.
- Single UI field for Gemini API Key (no project/location/model inputs).
- Robust call_gemini_text() that tries multiple client call shapes.
Notes:
 - Install: pip install streamlit pandas numpy google-generativeai
 - Provide API key in the UI or set GOOGLE_API_KEY env var.
 - Change MODEL_NAME constant if your account uses a different model identifier.
"""

import os, re, json, time
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np

# Attempt to import google.generativeai
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

# -------------------- Config --------------------
MODEL_NAME = os.getenv("GEMINI_MODEL", "models/text-bison-001")  # change if needed
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

# -------------------- Gemini (Vertex AI) resilient calls --------------------
def configure_genai(api_key: Optional[str] = None):
    """Configure google.generativeai client with an API key or rely on ADC otherwise."""
    if not GENAI_AVAILABLE:
        raise RuntimeError("google.generativeai package not installed. pip install google-generativeai")
    explicit_key = api_key or os.getenv("GOOGLE_API_KEY")
    if explicit_key:
        try:
            genai.configure(api_key=explicit_key)
        except Exception:
            # some client versions use a different configure signature; try a fallback
            try:
                genai.configure(project=None, location=None, api_key=explicit_key)
            except Exception as e:
                raise RuntimeError(f"genai.configure failed: {e}")
    else:
        # rely on ADC environment (GOOGLE_APPLICATION_CREDENTIALS)
        # no explicit configure call required in many setups
        pass

def call_gemini_text(prompt: str, model: str = MODEL_NAME, temperature: float = 0.0, max_output_tokens: int = 512) -> str:
    """
    Try multiple client interfaces in order to get the model text.
    Returns the raw text (best-effort).
    """
    if not GENAI_AVAILABLE:
        raise RuntimeError("google.generativeai not installed.")
    # Try chat.create
    try:
        if hasattr(genai, "chat") and hasattr(genai.chat, "create"):
            resp = genai.chat.create(model=model, messages=[{"role":"user","content":prompt}], temperature=temperature, max_output_tokens=max_output_tokens)
            # attempt to extract candidate text
            # Several client shapes exist; try common attributes
            if hasattr(resp, "candidates") and resp.candidates:
                candidate = resp.candidates[0]
                # candidate may have 'content' or 'message' fields
                content = getattr(candidate, "content", None) or getattr(candidate, "message", None) or str(candidate)
                return content if isinstance(content, str) else str(content)
            if hasattr(resp, "output"):
                return str(resp.output)
            # fallback
            return str(resp)
    except Exception:
        # continue to other attempts
        pass

    # Try generate_text
    try:
        if hasattr(genai, "generate_text"):
            resp = genai.generate_text(model=model, prompt=prompt, temperature=temperature, max_output_tokens=max_output_tokens)
            # Common shapes: dict with 'candidates', or object with .text
            if isinstance(resp, dict) and "candidates" in resp and resp["candidates"]:
                return resp["candidates"][0].get("output", "") or resp["candidates"][0].get("content","")
            if hasattr(resp, "text"):
                return getattr(resp, "text")
            return str(resp)
    except Exception:
        pass

    # Try genai.text.generate (another newer interface)
    try:
        if hasattr(genai, "text") and hasattr(genai.text, "generate"):
            resp = genai.text.generate(model=model, input=prompt, temperature=temperature, max_output_tokens=max_output_tokens)
            # example shapes: resp.text or resp.output or resp.candidates
            if hasattr(resp, "text"):
                return getattr(resp, "text")
            if hasattr(resp, "output"):
                return str(resp.output)
            return str(resp)
    except Exception:
        pass

    # Last-resort: call generic attribute or raise informative error
    raise RuntimeError("google.generativeai SDK present but no supported generation method found on this client. Inspect your client version and adapt the call shape.")

def gemini_embeddings(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    """Compute embeddings via genai client (best-effort across versions)."""
    if not GENAI_AVAILABLE:
        raise RuntimeError("google.generativeai not installed.")
    emb_model = model or "embedding-gecko-001"
    # Try genai.embeddings.create
    try:
        if hasattr(genai, "embeddings") and hasattr(genai.embeddings, "create"):
            resp = genai.embeddings.create(model=emb_model, input=texts)
            out = []
            for item in resp.data:
                out.append(item.embedding)
            return out
    except Exception:
        pass
    # Try genai.get_embeddings / genai.getEmbedding
    try:
        if hasattr(genai, "get_embeddings"):
            resp = genai.get_embeddings(model=emb_model, input=texts)
            return resp.get("embeddings") or resp.get("data")
    except Exception:
        pass
    raise RuntimeError("Embeddings not available via installed google.generativeai client. Check SDK docs and available embedding models.")

# -------------------- Utilities --------------------
def cosine_sim(a: List[float], b: List[float]) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Qforia-region (Gemini)", layout="wide")
st.title("Qforia-region — Query Fan-Out Simulator (Gemini)")

with st.sidebar:
    st.header("Configuration")
    st.write("Authentication: provide a Gemini API Key (or set GOOGLE_API_KEY env var).")
    api_key_box = st.text_input("Gemini API Key (optional)", type="password")
    st.markdown("---")
    st.number_of_queries = st.number_input("Number of synthetic queries (default)", min_value=5, max_value=200, value=DEFAULT_NUM_QUERIES)
    st.caption("Note: model name is taken from code constant MODEL_NAME. Change in the file if needed.")

# Main form (compact)
seed_query = st.text_input("Seed query", value="how to do call forwarding")
region_choice = st.selectbox("Region", [r["name"] for r in REGIONS], index=0)
region = next((r for r in REGIONS if r["name"] == region_choice), REGIONS[0])
num_queries = st.number_input("Number of synthetic queries", min_value=5, max_value=200, value=DEFAULT_NUM_QUERIES)
surface = st.selectbox("Target surface", AVAILABLE_SURFACES)
run_btn = st.button("Run Fan-Out 🚀")

# session state
if "last_df" not in st.session_state:
    st.session_state["last_df"] = pd.DataFrame()
if "embeddings" not in st.session_state:
    st.session_state["embeddings"] = []

if run_btn and seed_query.strip():
    try:
        configure_genai(api_key=api_key_box if api_key_box else None)
    except Exception as e:
        st.error(f"genai.configure error: {e}")
        raise

    prompt = make_prompt(seed_query, surface, region, int(num_queries), "")
    try:
        with st.spinner("Calling Gemini..."):
            raw = call_gemini_text(prompt, model=MODEL_NAME, temperature=0.0, max_output_tokens=1024)
    except Exception as e:
        st.error(f"Gemini call/config error: {e}")
        raw = None

    if raw:
        parsed = extract_json_array(raw)
        if parsed is None:
            st.error("Could not parse JSON from Gemini output. Showing raw output for debugging (first 2000 chars):")
            st.code(raw[:2000])
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

            # compute embeddings if possible
            try:
                texts = df["query"].tolist()
                emb = gemini_embeddings(texts)
                st.session_state["embeddings"] = emb
                st.success(f"Generated {len(df)} queries and computed embeddings.")
            except Exception as e:
                st.warning(f"Query generation succeeded but embeddings failed or are unavailable: {e}")
                st.session_state["embeddings"] = []

if not st.session_state["last_df"].empty:
    df = st.session_state["last_df"]
    st.subheader("Synthetic Queries")
    st.dataframe(df)

    st.markdown("---")
    st.subheader("Passage -> Query (embeddings)")
    passage = st.text_area("Paste page passage (optional)")
    match_btn = st.button("Match passage to generated queries")

    if match_btn and passage.strip():
        if not st.session_state["embeddings"]:
            st.error("No embeddings available. Re-run and ensure embedding-capable model is available.")
        else:
            try:
                p_emb = gemini_embeddings([passage])[0]
                scores = []
                for i, q_emb in enumerate(st.session_state["embeddings"]):
                    sim = cosine_sim(p_emb, q_emb)
                    scores.append((df.iloc[i]["query"], sim, df.iloc[i]["intent"], df.iloc[i]["rationale"]))
                scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
                out = [{"query": s[0], "score": round(s[1],4), "intent": s[2], "rationale": s[3]} for s in scores_sorted[:20]]
                st.table(pd.DataFrame(out))
            except Exception as e:
                st.error(f"Embeddings matching failed: {e}")

st.markdown("---")
st.caption("This app accepts a Gemini API key. If your client version differs, adapt the call_gemini_text() branches. Region hints bias outputs but do not reproduce Google internal fan-outs.")
