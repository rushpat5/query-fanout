import os
import streamlit as st
import google.generativeai as genai
import pandas as pd
import json

st.set_page_config(page_title="Qforia-free — Query Fan-Out Simulator", layout="wide")

st.title("Qforia-free — Query Fan-Out Simulator (Gemini-Flash)")

# ----------------------------
# API KEY
# ----------------------------
api_key = st.sidebar.text_input("Gemini API Key (free tier)", type="password")
if api_key:
    genai.configure(api_key=api_key)

# ----------------------------
# User Inputs
# ----------------------------
seed_query = st.text_input("Seed query", "how to do call forwarding")
num_queries = st.number_input("How many synthetic queries?", 15, 200, 30)
extra = st.text_area("Extra instructions (optional)")

run = st.button("Run Fan-Out (Flash)")

# ----------------------------
# Prompt Template (forced OG Qforia schema)
# ----------------------------

def build_prompt(q, n, extra):
    return f"""
You are Qforia. You generate synthetic queries using the classic Qforia schema.

For the seed query: "{q}"
Generate exactly {n} synthetic search queries.

You MUST return valid JSON with this format:

{{
  "items": [
    {{
      "lookup_query": "...",
      "query": "...",
      "type": "reformulation | implicit | entity_expansion | comparative | related | personalized",
      "user_intent": "...",
      "reasoning": "...",
      "routing_format": "how_to_steps | glossary/definition | faq_page | comparison_table | forum/qna | tutorial_video/transcript",
      "format_reason": "..."
    }},
    ...
  ]
}}

Rules:
- No markdown.
- No code fences.
- JSON only.
- lookup_query must equal the seed query.
- Every query must be unique.
- Provide realistic reasoning and routing_format.

Additional instructions (optional):
{extra}
    """


# ----------------------------
# Run Fan-Out
# ----------------------------

if run and api_key:
    prompt = build_prompt(seed_query, num_queries, extra)

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")  # free-tier model
        response = model.generate_content(prompt)

        raw = response.text

        # Try parsing JSON
        try:
            data = json.loads(raw)
        except:
            st.error("Model response could not be parsed as JSON.")
            st.text(raw)
            st.stop()

        items = data.get("items", [])
        if not items:
            st.error("JSON parsed, but no items found.")
            st.text(raw)
            st.stop()

        df = pd.DataFrame(items)
        st.success(f"Generated {len(df)} queries.")

        st.subheader("Synthetic Queries (with routing format)")
        st.dataframe(df, use_container_width=True)

        # Export buttons
        st.download_button("Export to CSV", df.to_csv(index=False), "qforia_free.csv")
        st.download_button("Export to JSON", json.dumps(items, indent=2), "qforia_free.json")

    except Exception as e:
        st.error(str(e))
        st.stop()
