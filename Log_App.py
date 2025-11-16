import streamlit as st
import requests
import json

# -----------------------------
# Constants
# -----------------------------
FLASH_MODEL = "gemini-2.0-flash"
BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{FLASH_MODEL}:generateContent"


# -----------------------------
# REST CALL FUNCTION
# -----------------------------
def call_gemini_flash(api_key: str, prompt: str):
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        response = requests.post(
            BASE_URL,
            params={"key": api_key},
            headers=headers,
            json=payload,
            timeout=40
        )
        return response.status_code, response.text

    except Exception as e:
        return -1, str(e)


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("Qforia-Free — Query Fan-Out Simulator (Flash Model Only)")

st.subheader("Authentication")
api_key = st.text_input("Gemini API Key (free-tier)", type="password")

st.subheader("Fan-out Settings")
seed_query = st.text_input("Seed Query", "how to do call forwarding")
region = st.selectbox("Region (bias via prompt only)", [
    "United States", "United Kingdom", "Canada", "India",
    "Australia", "Singapore", "Germany", "Brazil", "Global"
])
num = st.number_input("Number of synthetic queries", min_value=5, max_value=200, value=30)

extra_instr = st.text_area("Extra instructions (optional)", "")

run = st.button("Run Fan-Out (Flash Model)")


# -----------------------------
# RUN FAN OUT
# -----------------------------
if run:
    if not api_key.strip():
        st.error("API key required.")
    else:
        st.info("Generating via free-tier flash model...")

        # Prompt engineering for "region bias"
        fanout_prompt = f"""
You are generating synthetic search queries for SEO evaluation.

Seed query:
{seed_query}

Region:
{region}

Number of synthetic queries:
{num}

Instructions:
- Output EXACTLY a JSON list (array) of strings.
- No explanation, no prose.
- Each query must reflect REALISTIC user intent patterns for the region.
- Avoid duplicates.
- Avoid unnatural rewrites.
- Preserve plausible search traffic behavior.
- Additional instructions (if any): {extra_instr}
        """

        status, body = call_gemini_flash(api_key, fanout_prompt)

        st.subheader("Raw Response")
        st.code(body)

        if status != 200:
            st.error(f"Flash-model REST call failed. Status: {status}")
        else:
            try:
                parsed = json.loads(body)["candidates"][0]["content"]["parts"][0]["text"]
                final_json = json.loads(parsed)
                st.success("Parsed synthetic queries:")
                st.write(final_json)

                # Export options
                st.download_button("Download JSON", data=json.dumps(final_json, indent=2), file_name="fanout.json")

            except Exception:
                st.warning("Model response could not be parsed as JSON. Showing raw output above.")


# -----------------------------
# Notes
# -----------------------------
st.markdown("""
---

### Notes
- This tool uses **free-tier Gemini Flash only**.
- Accuracy is lower than Qforia’s original PRO model.
- Region effects are prompt-based only (no proprietary signals).
- No embeddings, no Vertex, no client-library reflection, no deprecated endpoints.

""")
