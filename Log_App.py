import streamlit as st
import pandas as pd
import requests
import json
import re

# ---------- Configuration ----------
FLASH_MODEL = "models/gemini-2.5-flash"
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{FLASH_MODEL}:generateContent"

def build_prompt(seed, region, n, extra=""):
    ts = pd.Timestamp.utcnow().isoformat() + "Z"
    return f"""
You are an assistant that MUST OUTPUT valid JSON only: a JSON array of exactly {n} objects.

Each object MUST have:
- query
- intent (informational|commercial|transactional|navigational|investigational)
- entity
- variation_type (paraphrase|narrowing|expansion|entity-focus|question-form|long-tail|comparative)
- rationale

Seed query: "{seed}"
Region: "{region}"
Timestamp: {ts}

Extra instructions:
{extra}

Output ONLY the JSON array. Nothing else.
""".strip()

def extract_json_array(text):
    try:
        return json.loads(text)
    except:
        m = re.search(r"\[[\s\S]*\]", text)
        if m:
            try:
                return json.loads(m.group(0))
            except:
                return None
    return None

def call_flash_model(prompt, api_key):
    payload = {
        "contents": [
            {"parts":[{"text": prompt}]}
        ]
    }
    try:
        r = requests.post(
            ENDPOINT,
            params={"key": api_key},
            json=payload,
            timeout=40
        )
    except Exception as e:
        return False, None, [{"url": ENDPOINT, "status": None, "body": repr(e)}]

    attempt = {"url": ENDPOINT, "status": r.status_code, "body": r.text[:600]}
    if r.status_code == 200:
        try:
            data = r.json()
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text","") for p in parts)
            return True, text, [attempt]
        except:
            return True, r.text, [attempt]
    else:
        return False, None, [attempt]

# ---------- Streamlit UI ----------
st.set_page_config(layout="wide", page_title="Qforia Flash-Only")
st.title("Qforia — Free-Tier Flash Model Fan-Out")

with st.sidebar:
    st.header("Authentication")
    api_key = st.text_input("Gemini API Key (free tier)", type="password")

seed = st.text_input("Seed query", "how to do call forwarding")
region = st.selectbox("Region", ["United States","United Kingdom","India","Canada","Australia"], index=0)
n = st.number_input("Number of synthetic queries", min_value=3, max_value=200, value=30)
extra = st.text_area("Extra instructions (optional)", "")

run = st.button("Run Fan-Out (flash model)")

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
        prompt = build_prompt(seed, region, n, extra)
        ok, raw, attempts = call_flash_model(prompt, api_key.strip())
        st.session_state.attempts = attempts

        if not ok:
            st.error("Flash-model REST call failed.")
        else:
            st.success("Flash model generation succeeded.")
            st.session_state.raw = raw
            parsed = extract_json_array(raw)
            if parsed is None:
                st.warning("Could not parse JSON. Raw output:")
                st.code(raw[:4000])
            else:
                rows = []
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

if not st.session_state.df.empty:
    st.subheader("Synthetic Queries")
    st.dataframe(st.session_state.df)
    st.download_button("Download CSV",
                       st.session_state.df.to_csv(index=False).encode("utf-8"),
                       file_name="qforia_flash.csv", mime="text/csv")
    st.download_button("Download JSON",
                       st.session_state.df.to_json(orient="records", force_ascii=False),
                       file_name="qforia_flash.json", mime="application/json")

if st.session_state.raw:
    st.subheader("Raw Output (truncated)")
    st.code(st.session_state.raw[:3000])

if st.session_state.attempts:
    st.subheader("REST Attempt Log")
    st.dataframe(pd.DataFrame(st.session_state.attempts))
