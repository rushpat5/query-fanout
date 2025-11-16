# app.py
import streamlit as st
import pandas as pd
import requests
import json
import re

# -----------------------------------
# Configuration
# -----------------------------------
DEFAULT_MODEL = "models/text-bison-001"
DEFAULT_NUM = 30

GENERATIVE_BASES = [
    "https://generativelanguage.googleapis.com/v1",
    "https://generativelanguage.googleapis.com/v1beta2",
]

MODEL_PROBES = [
    "models/gemini-1.5-pro",
    "models/gemini-1.5",
    "models/gemini-1.0",
    "models/text-bison-001",
    "models/chat-bison-001",
]

# -----------------------------------
# Prompt builder
# -----------------------------------
def make_prompt(seed, region, n, extra=""):
    ts = pd.Timestamp.utcnow().isoformat() + "Z"
    return f"""
You are an assistant that MUST OUTPUT valid JSON only: a JSON array of exactly {n} objects.

Each object must have:
- query
- intent (informational|commercial|transactional|navigational|investigational)
- entity
- variation_type (paraphrase|narrowing|expansion|entity-focus|question-form|long-tail|comparative)
- rationale (one sentence)

Seed query: "{seed}"
Region: "{region}"
Timestamp: {ts}

Extra instructions:
{extra}

Output ONLY the JSON array. Nothing else.
""".strip()

# -----------------------------------
# JSON extraction
# -----------------------------------
def extract_json_array(text):
    try:
        return json.loads(text)
    except:
        m = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except:
                return None
    return None

# -----------------------------------
# REST generator
# -----------------------------------
def generate_via_rest(prompt, api_key, model):
    attempts = []

    model_list = [model] if model else MODEL_PROBES

    for base in GENERATIVE_BASES:
        for m in model_list:

            url = f"{base}/{m}:generateText"

            payload = {
                "prompt": {"text": prompt},
                "maxOutputTokens": 1200,
                "temperature": 0.0
            }

            try:
                r = requests.post(
                    url,
                    params={"key": api_key},
                    json=payload,
                    timeout=30
                )
                attempts.append({
                    "url": url,
                    "status": r.status_code,
                    "body": r.text[:600]
                })

                if r.status_code == 200:
                    data = r.json()

                    # direct candidate structure
                    if isinstance(data, dict):
                        if "candidates" in data and data["candidates"]:
                            cand = data["candidates"][0]
                            for k in ("output","content","text","displayText"):
                                if isinstance(cand.get(k), str):
                                    return True, cand[k], attempts

                        # fallback to any known top-level fields
                        for k in ("output","text","content","result","response"):
                            if k in data and isinstance(data[k], str):
                                return True, data[k], attempts

                    # fallback raw
                    return True, r.text, attempts

            except Exception as e:
                attempts.append({"url": url, "status": None, "body": repr(e)})

    return False, None, attempts

# -----------------------------------
# UI
# -----------------------------------
st.set_page_config(layout="wide", page_title="Qforia REST-only")

st.title("Qforia — REST-only Gemini Fan-Out")

with st.sidebar:
    st.header("Auth & Model")
    api_key = st.text_input("Gemini API Key (required)", type="password")
    model_resource = st.text_input("Model resource (optional)", value=DEFAULT_MODEL)
    st.caption("REST does not depend on google-generativeai client.")

seed = st.text_input("Seed query", "how to do call forwarding")
region = st.selectbox("Region", ["United States","United Kingdom","India","Canada","Australia"], index=0)
num = st.number_input("Number of queries", min_value=3, max_value=200, value=DEFAULT_NUM)
extra = st.text_area("Extra prompt instructions")

run = st.button("Run Fan-Out (REST)")


if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "raw" not in st.session_state:
    st.session_state.raw = ""
if "attempts" not in st.session_state:
    st.session_state.attempts = []

if run:
    if not api_key:
        st.error("API key required.")
    else:
        prompt = make_prompt(seed, region, int(num), extra)

        ok, raw, attempts = generate_via_rest(
            prompt,
            api_key.strip(),
            model_resource.strip() if model_resource else None
        )

        st.session_state.attempts = attempts

        if not ok:
            st.error("REST generation failed. See attempts below.")
            st.dataframe(pd.DataFrame(attempts))
        else:
            st.success("REST generation succeeded.")
            st.session_state.raw = raw

            parsed = extract_json_array(raw)
            if parsed is None:
                st.warning("Could not parse JSON. Raw output:")
                st.code(raw[:4000])
            else:
                rows=[]
                for i, item in enumerate(parsed):
                    rows.append({
                        "rank": i+1,
                        "query": item.get("query",""),
                        "intent": item.get("intent",""),
                        "entity": item.get("entity",""),
                        "variation_type": item.get("variation_type",""),
                        "rationale": item.get("rationale",""),
                    })
                st.session_state.df = pd.DataFrame(rows)

# results
if not st.session_state.df.empty:
    st.subheader("Synthetic Queries")
    st.dataframe(st.session_state.df)

    st.download_button(
        "Download CSV",
        st.session_state.df.to_csv(index=False).encode("utf-8"),
        file_name="qforia_rest.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download JSON",
        st.session_state.df.to_json(orient="records", force_ascii=False),
        file_name="qforia_rest.json",
        mime="application/json"
    )

if st.session_state.raw:
    st.markdown("---")
    st.subheader("Raw Output (truncated)")
    st.code(st.session_state.raw[:4000])

if st.session_state.attempts:
    st.markdown("---")
    st.subheader("REST Attempts")
    st.dataframe(pd.DataFrame(st.session_state.attempts))
