# app.py
"""
Defensive Qforia-approx Streamlit app that tries:
 - public Generative Language REST endpoints (v1, v1beta2) with several model-name variants
 - optional Vertex-style project-scoped REST endpoint if user supplies project_id/location/model_id
The app requires your API key (Gemini/Generative) and a model identifier or will attempt common variants.
It always surfaces raw HTTP responses for debugging.
"""

import os, json, re, time
from typing import Optional, List, Dict, Any
import requests
import streamlit as st
import pandas as pd

# --- config: candidate model variants to try against generativelanguage endpoints
GEN_API_BASES = [
    "https://generativelanguage.googleapis.com/v1",
    "https://generativelanguage.googleapis.com/v1beta2",
]
MODEL_VARIANTS = [
    "models/gemini-1.5-pro","gemini-1.5-pro",
    "models/gemini-1.5","gemini-1.5",
    "models/gemini-1.0","gemini-1.0",
    "models/text-bison-001","text-bison-001",
    "models/chat-bison-001","chat-bison-001",
]

VERTEX_BASE_TEMPLATE = "https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/models/{model}:predict"

# --- UI
st.set_page_config(layout="wide", page_title="Qforia-API-key defensive")
st.title("Qforia-API-key — defensive generation (shows raw responses)")

with st.sidebar:
    st.header("Minimal inputs")
    st.write("Provide Gemini/Generative API key and (preferably) a model resource. Optional: project & location for Vertex-style endpoint attempts.")
    api_key = st.text_input("Gemini API Key (required)", type="password")
    model_input = st.text_input("Model resource (optional) e.g. models/text-bison-001 or gemini-1.5-pro")
    st.markdown("---")
    st.write("Optional: if your model is a Vertex model in a project, provide these (used only if public attempts 404).")
    project_id = st.text_input("Project ID (optional)")
    location = st.text_input("Location (optional) e.g. us-central1")
    st.caption("If you do not know which to use, leave project/location blank; the app will try public endpoints first and show raw failures.")

seed_query = st.text_input("Seed query", value="how to do call forwarding")
region = st.selectbox("Region hint", ["United States","United Kingdom","India","Canada","Australia"])
num_queries = st.number_input("Number of synthetic queries", min_value=3, max_value=200, value=20)
extra_instructions = st.text_area("Extra prompt instructions (optional)", value="")
run = st.button("Run Fan-Out (attempt)")

# --- helpers
def build_prompt(seed: str, region: str, n: int, extra: str=""):
    ts = pd.Timestamp.utcnow().isoformat() + "Z"
    prompt = f"""
You are an assistant that MUST OUTPUT valid JSON ONLY: a JSON array of exactly {n} objects.
Each object must contain: query, intent (informational|commercial|transactional|navigational|investigational), entity, variation_type, rationale.

Seed: "{seed}"
Region hint: "{region}"
Timestamp: {ts}

Extra: {extra}

Output the JSON array and nothing else.
"""
    return prompt.strip()

def call_generative(base: str, model: str, prompt: str, api_key: str, timeout=60):
    url = f"{base.rstrip('/')}/{model}:generateText"
    params = {"key": api_key}
    payload = {"prompt": {"text": prompt}, "maxOutputTokens": 1024, "temperature": 0.0}
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.post(url, params=params, json=payload, headers=headers, timeout=timeout)
    except Exception as e:
        return {"ok": False, "status": None, "body": f"request exception: {e}", "url": url}
    return {"ok": r.status_code == 200, "status": r.status_code, "body": r.text, "url": url}

def call_vertex_predict(project: str, location: str, model: str, prompt: str, api_key: str, timeout=60):
    # This attempts a Vertex predict-like call using API key in URL if possible.
    # NOTE: depending on your model, the payload shape may differ. We send a conservative 'instances' shape.
    url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/models/{model}:predict"
    params = {"key": api_key}
    payload = {"instances": [{"content": prompt}], "parameters": {"maxOutputTokens": 1024}}
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.post(url, params=params, json=payload, headers=headers, timeout=timeout)
    except Exception as e:
        return {"ok": False, "status": None, "body": f"request exception: {e}", "url": url}
    return {"ok": r.status_code in (200,201), "status": r.status_code, "body": r.text, "url": url}

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

# --- run logic
if run:
    if not api_key or not api_key.strip():
        st.error("API key required.")
    else:
        prompt = build_prompt(seed_query, region, int(num_queries), extra_instructions)
        st.info("Trying public Generative Language endpoints with candidate model variants...")
        attempts = []
        success = False
        result_body = None
        used_base = None
        used_model = None

        # Try user-provided model first if given (both variant forms)
        variant_list = []
        if model_input and model_input.strip():
            variant_list.append(model_input.strip())
            if model_input.strip().startswith("models/"):
                variant_list.append(model_input.strip().replace("models/","",1))
            else:
                variant_list.append("models/" + model_input.strip())
        # then default variants
        for v in MODEL_VARIANTS:
            if v not in variant_list:
                variant_list.append(v)

        for base in GEN_API_BASES:
            if success: break
            for model in variant_list:
                r = call_generative(base, model, prompt, api_key.strip())
                attempts.append({"path": r["url"], "model": model, "base": base, "status": r["status"], "ok": r["ok"], "short": (r["body"] or "")[:400]})
                if r["ok"]:
                    success = True
                    result_body = r["body"]
                    used_base = base
                    used_model = model
                    break

        # If public attempts all 404 and user provided project/location/model, try Vertex-style predict
        if (not success) and project_id and location and model_input:
            st.info("Public attempts failed. Trying Vertex-style predict endpoint (project-scoped) with provided project/location/model.")
            rv = call_vertex_predict(project_id.strip(), location.strip(), model_input.strip(), prompt, api_key.strip())
            attempts.append({"path": rv["url"], "model": model_input.strip(), "base": "vertex_predict", "status": rv["status"], "ok": rv["ok"], "short": (rv["body"] or "")[:400]})
            if rv["ok"]:
                success = True
                result_body = rv["body"]
                used_base = "vertex_predict"
                used_model = model_input.strip()

        # Show attempts summary
        st.markdown("### Attempt summary")
        df_attempts = pd.DataFrame(attempts)
        st.dataframe(df_attempts)

        if not success:
            st.error("All attempts failed. Inspect the attempt summary for exact HTTP codes and server messages.")
        else:
            st.success(f"Success with model '{used_model}' at base '{used_base}'. Showing parsed JSON if available.")
            st.markdown("**Raw response (truncated)**")
            st.code((result_body or "")[:4000])
            parsed = extract_json_array(result_body or "")
            if parsed is None:
                st.warning("Could not parse strict JSON array from the model output. Raw output shown above.")
            else:
                # show table of candidate queries
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
                st.subheader("Synthetic queries (parsed)")
                st.dataframe(pd.DataFrame(rows))
