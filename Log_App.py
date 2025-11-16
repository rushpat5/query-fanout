# app.py
"""
Streamlit app: Qforia-client-style (Gemini) — uses google.generativeai client library.
- Enter your Gemini API key (UI).
- Enter a model resource (recommended) or use the default.
- The app uses the client library (preferred) rather than raw REST calls.
"""

import os
import json
import re
from typing import Optional, List, Dict, Any

import streamlit as st
import pandas as pd

# Attempt import of the official client library
try:
    import google.generativeai as genai  # type: ignore
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

# ----------------- Configuration -----------------
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "models/text-bison-001")  # change to a model you know
DEFAULT_NUM = 30

# ----------------- Helpers -----------------
def make_prompt(seed: str, region_hint: str, n: int, extra: str = "") -> str:
    ts = pd.Timestamp.utcnow().isoformat() + "Z"
    prompt = f"""
You are a utility assistant that MUST OUTPUT valid JSON only: a JSON array of exactly {n} objects.

Each object must have:
 - query (string)
 - intent (string): informational, commercial, transactional, navigational, investigational
 - entity (string)
 - variation_type (string): paraphrase, narrowing, expansion, entity-focus, question-form, long-tail, comparative
 - rationale (string): one-sentence justification

Seed query: "{seed}"
Region hint: "{region_hint}"
Timestamp: {ts}

Additional instructions:
{extra}

Produce the JSON array and nothing else.
"""
    return prompt.strip()

def extract_json_array(text: str) -> Optional[List[Dict[str,Any]]]:
    try:
        return json.loads(text)
    except Exception:
        # attempt to find the first JSON array in the text
        m = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None

def client_generate_text(prompt: str, model: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
    """
    Try multiple client shapes. Return the textual output (best-effort).
    Raises RuntimeError with helpful message on failure.
    """
    if not GENAI_AVAILABLE:
        raise RuntimeError("google.generativeai client not installed. Run: pip install google-generativeai")

    # Prefer genai.generate_text if available
    try:
        if hasattr(genai, "generate_text"):
            resp = genai.generate_text(model=model, prompt=prompt, max_output_tokens=max_tokens, temperature=temperature)
            # resp may be an object or dict
            if isinstance(resp, dict) and "candidates" in resp and resp["candidates"]:
                cand = resp["candidates"][0]
                return cand.get("output") or cand.get("content") or str(cand)
            # object with .text
            if hasattr(resp, "text"):
                return getattr(resp, "text")
            # fallback to str
            return str(resp)
    except Exception as e:
        # continue to next shape, but keep last exception for debugging
        last_exc = e

    # Try genai.chat.create if exists
    try:
        if hasattr(genai, "chat") and hasattr(genai.chat, "create"):
            chat_resp = genai.chat.create(model=model, messages=[{"role":"user","content":prompt}], temperature=temperature, max_output_tokens=max_tokens)
            if hasattr(chat_resp, "candidates") and chat_resp.candidates:
                cand = chat_resp.candidates[0]
                return getattr(cand, "content", getattr(cand, "message", str(cand)))
            if hasattr(chat_resp, "output"):
                return getattr(chat_resp, "output")
            return str(chat_resp)
    except Exception as e:
        last_exc = e

    # Try genai.text.generate (another interface)
    try:
        if hasattr(genai, "text") and hasattr(genai.text, "generate"):
            resp = genai.text.generate(model=model, input=prompt, temperature=temperature, max_output_tokens=max_tokens)
            if hasattr(resp, "text"):
                return getattr(resp, "text")
            return str(resp)
    except Exception as e:
        last_exc = e

    # If we reached here, raise a clear error with last exception info
    raise RuntimeError(f"Client present but no supported generation method succeeded. Last error: {repr(last_exc)}")

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Qforia-client (Gemini)", layout="wide")
st.title("Qforia-client-style — Query Fan-Out (Gemini)")

with st.sidebar:
    st.header("Authentication")
    st.write("Paste your Gemini API key. The app will configure the official client library with this key.")
    api_key = st.text_input("Gemini API Key", type="password")
    st.markdown("---")
    st.write("Model resource (recommended) — example: models/text-bison-001 or models/gemini-1.5-pro")
    model_resource = st.text_input("Model resource", value=DEFAULT_MODEL)
    st.markdown("---")
    st.write("Generation controls")
    num_queries = st.number_input("Number of synthetic queries", min_value=3, max_value=200, value=DEFAULT_NUM)
    region_hint = st.selectbox("Region hint", ["United States", "United Kingdom", "India", "Canada", "Australia"], index=0)

col1, col2 = st.columns([3,1])
with col1:
    seed_query = st.text_input("Seed query", value="how to do call forwarding")
    extra_instructions = st.text_area("Extra prompt instructions (optional)", value="")
    run_btn = st.button("Run Fan-Out (client)")

with col2:
    st.write("Export")
    export_csv = st.button("Export last results to CSV")
    export_json = st.button("Export last results to JSON")

# session state
if "last_df" not in st.session_state:
    st.session_state["last_df"] = pd.DataFrame()
if "raw" not in st.session_state:
    st.session_state["raw"] = ""

# Run generation
if run_btn:
    if not api_key or not api_key.strip():
        st.error("Gemini API key is required.")
    elif not model_resource or not model_resource.strip():
        st.error("Model resource is required (enter a model name your key can access).")
    else:
        # configure client with key
        try:
            genai.configure(api_key=api_key.strip())
        except Exception:
            # some client versions accept genai.configure(api_key=...) directly; if it fails, still continue
            try:
                genai.configure(project=None, location=None, api_key=api_key.strip())
            except Exception as e:
                st.error(f"Failed to configure google.generativeai client: {e}")
                raise

        prompt = make_prompt(seed_query, region_hint, int(num_queries), extra_instructions)
        st.info("Calling Gemini via client library...")
        try:
            text_out = client_generate_text(prompt, model_resource.strip(), max_tokens=1200, temperature=0.0)
        except Exception as e:
            st.error(f"Generation failed (client): {e}")
            st.code(str(e))
            text_out = None

        if text_out:
            st.session_state["raw"] = text_out
            parsed = extract_json_array(text_out)
            if parsed is None:
                st.warning("Could not parse JSON array from model output. Showing raw output for debugging.")
                st.code(text_out[:4000])
            else:
                rows = []
                for i, item in enumerate(parsed):
                    rows.append({
                        "rank": i+1,
                        "query": item.get("query",""),
                        "intent": item.get("intent",""),
                        "entity": item.get("entity",""),
                        "variation_type": item.get("variation_type",""),
                        "rationale": item.get("rationale","")
                    })
                df = pd.DataFrame(rows)
                st.session_state["last_df"] = df
                st.success(f"Generated {len(df)} items.")

# show results if available
if not st.session_state["last_df"].empty:
    st.subheader("Synthetic Queries")
    st.dataframe(st.session_state["last_df"])

    if export_csv:
        st.download_button("Download CSV", data=st.session_state["last_df"].to_csv(index=False).encode("utf-8"), file_name="qforia_client_results.csv", mime="text/csv")
    if export_json:
        st.download_button("Download JSON", data=st.session_state["last_df"].to_json(orient="records", force_ascii=False), file_name="qforia_client_results.json", mime="application/json")

    st.markdown("---")
    st.subheader("Raw model output (truncated)")
    st.code(st.session_state["raw"][:4000])

# If client library missing, show install hint
if not GENAI_AVAILABLE:
    st.error("google-generativeai client library not installed. Install with: pip install google-generativeai")
    st.stop()
