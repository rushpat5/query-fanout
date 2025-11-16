import streamlit as st
import pandas as pd
import requests
import json
import re

# -----------------------------------------------------
# Correct AND supported Gemini REST endpoint
# -----------------------------------------------------
GEMINI_ENDPOINTS = [
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent",
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.0-pro:generateContent",
]


def build_prompt(seed, region, n, extra):
    return f"""
You are an assistant generating synthetic user queries. Output ONLY valid JSON.
Return an array of {n} objects.

Each object MUST have:
- query
- intent
- entity
- variation_type
- rationale

Seed query: "{seed}"
Region: "{region}"

Extra rules:
{extra}

Output ONLY a JSON array. No explanation, no markdown.
""".strip()


def extract_json_array(txt):
    try:
        return json.loads(txt)
    except:
        m = re.search(r"\[[\s\S]*\]", txt)
        if m:
            try:
                return json.loads(m.group(0))
            except:
                return None
    return None


def call_gemini_rest(prompt, api_key):
    attempts = []

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    for url in GEMINI_ENDPOINTS:
        try:
            r = requests.post(
                url,
                params={"key": api_key},
                json=payload,
                timeout=40
            )

            attempts.append({
                "url": url,
                "status": r.status_code,
                "body": r.text[:600]
            })

            if r.status_code == 200:
                data = r.json()

                # Gemini outputs text inside candidates[0].content.parts[*].text
                try:
                    parts = data["candidates"][0]["content"]["parts"]
                    all_text = "".join(p.get("text", "") for p in parts)
                    return True, all_text, attempts
                except:
                    return True, r.text, attempts

        except Exception as e:
            attempts.append({"url": url, "status": None, "body": repr(e)})

    return False, None, attempts


# -----------------------------------------------------
# Streamlit UI
# -----------------------------------------------------
st.set_page_config(layout="wide", page_title="Qforia (Gemini REST)")

st.title("Qforia — High-Fidelity Query Fan-Out (Gemini REST API)")

with st.sidebar:
    st.header("Authentication")
    api_key = st.text_input("Gemini API Key", type="password")

seed = st.text_input("Seed query", "how to do call forwarding")

region = st.selectbox("Region", [
    "United States",
    "United Kingdom",
    "India",
    "Canada",
    "Australia",
])

n = st.number_input("Number of synthetic queries", 5, 200, 30)
extra = st.text_area("Extra instructions (optional)", "")

run = st.button("Run Fan-Out")

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

        ok, raw, attempts = call_gemini_rest(prompt, api_key)

        st.session_state.attempts = attempts

        if not ok:
            st.error("REST generation failed. See attempts.")
        else:
            st.success("Gemini generation completed.")
            st.session_state.raw = raw

            parsed = extract_json_array(raw)
            if parsed is None:
                st.warning("Could not parse JSON.")
                st.code(raw[:4000])
            else:
                rows = []
                for i, item in enumerate(parsed):
                    rows.append({
                        "rank": i+1,
                        "query": item.get("query", ""),
                        "intent": item.get("intent", ""),
                        "entity": item.get("entity", ""),
                        "variation_type": item.get("variation_type", ""),
                        "rationale": item.get("rationale", ""),
                    })
                st.session_state.df = pd.DataFrame(rows)


if not st.session_state.df.empty:
    st.subheader("Generated Queries")
    st.dataframe(st.session_state.df)

    st.download_button(
        "Download CSV",
        st.session_state.df.to_csv(index=False).encode("utf-8"),
        file_name="qforia_gemini.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download JSON",
        st.session_state.df.to_json(orient="records"),
        file_name="qforia_gemini.json",
        mime="application/json"
    )


if st.session_state.raw:
    st.subheader("Raw Output (truncated)")
    st.code(st.session_state.raw[:3000])

if st.session_state.attempts:
    st.subheader("Attempt Log")
    st.dataframe(pd.DataFrame(st.session_state.attempts))
