# app.py
"""
Robust Qforia-client app: attempts dynamic client invocation via reflection, then REST fallback.
Instructions:
 - pip install streamlit pandas requests
 - If you want to use the official client path, also pip install google-generativeai (any version)
 - Run: streamlit run app.py
"""
import os
import json
import re
import inspect
from typing import Optional, List, Dict, Any
import requests
import streamlit as st
import pandas as pd

# Try import of the client (may exist with various shapes)
try:
    import google.generativeai as genai  # type: ignore
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

# Config
GENERATIVE_BASES = [
    "https://generativelanguage.googleapis.com/v1",
    "https://generativelanguage.googleapis.com/v1beta2",
]
MODEL_PROBES = [
    "models/gemini-1.5-pro", "gemini-1.5-pro",
    "models/gemini-1.5", "gemini-1.5",
    "models/gemini-1.0", "gemini-1.0",
    "models/text-bison-001", "text-bison-001",
    "models/chat-bison-001", "chat-bison-001",
]
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "models/text-bison-001")
DEFAULT_NUM = 30

# ---------------- helpers ----------------
def make_prompt(seed: str, region: str, n: int, extra: str="") -> str:
    ts = pd.Timestamp.utcnow().isoformat() + "Z"
    return f"""
You are an assistant that MUST OUTPUT valid JSON only: a JSON array of exactly {n} objects.
Each object must have: query, intent (informational|commercial|transactional|navigational|investigational), entity, variation_type, rationale.
Seed query: "{seed}"
Region hint: "{region}"
Timestamp: {ts}
Extra: {extra}
Output the JSON array and nothing else.
""".strip()

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

# Reflection-based client invoker
def call_via_client_reflection(prompt: str, model: str, max_tokens:int=1024, temperature:float=0.0, debug_limit:int=4000) -> Dict[str,Any]:
    """
    Inspect the genai module and try plausible callables with guessed kwargs.
    Returns dict: {"ok":bool,"method":str,"result":str,"attempts":[...],"last_exception":str}
    """
    attempts = []
    if not GENAI_AVAILABLE:
        return {"ok": False, "method": None, "result": None, "attempts": attempts, "last_exception": "client_not_installed"}

    # Candidate attribute names to search for (broad)
    candidate_names = []
    # Top-level names in genai
    for name in dir(genai):
        lname = name.lower()
        if any(tok in lname for tok in ("generate","create","chat","text","predict","completion","model","response")):
            candidate_names.append(name)
    # Also search genai.chat if present
    if hasattr(genai, "chat"):
        for name in dir(getattr(genai, "chat")):
            if name.startswith("_"):
                continue
            lname = name.lower()
            if any(tok in lname for tok in ("create","generate","response","send")):
                candidate_names.append(f"chat.{name}")
    # de-duplicate preserving order
    seen = set()
    ordered = []
    for n in candidate_names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)

    # Build a list of candidate callables (callable obj, display name)
    callables = []
    for name in ordered:
        try:
            if "." in name:
                parent, attr = name.split(".",1)
                obj = getattr(getattr(genai, parent), attr)
            else:
                obj = getattr(genai, name)
            if callable(obj):
                callables.append((obj, name))
        except Exception:
            continue

    # known param name permutations to try
    param_variants = [
        {"model": model, "prompt": prompt, "max_output_tokens": max_tokens, "temperature": temperature},
        {"model": model, "input": prompt, "max_output_tokens": max_tokens, "temperature": temperature},
        {"model": model, "prompt": prompt, "maxOutputTokens": max_tokens, "temperature": temperature},
        {"model": model, "content": prompt, "max_output_tokens": max_tokens, "temperature": temperature},
        {"model": model, "messages": [{"role":"user","content":prompt}], "max_output_tokens": max_tokens, "temperature": temperature},
        {"model": model, "messages": [{"role":"user","content":prompt}], "maxTokens": max_tokens, "temperature": temperature},
        {"model": model, "prompt": {"text": prompt}, "max_output_tokens": max_tokens, "temperature": temperature},
        {"model": model, "text": prompt, "max_output_tokens": max_tokens, "temperature": temperature},
        {"model": model, "input": {"text": prompt}, "max_output_tokens": max_tokens, "temperature": temperature},
    ]

    last_exc = None
    for (fn, name) in callables:
        sig = None
        try:
            sig = inspect.signature(fn)
        except Exception:
            sig = None
        tried = []
        for pv in param_variants:
            # filter pv keys to parameters fn accepts where possible
            if sig is not None:
                kwargs = {}
                for k,v in pv.items():
                    if k in sig.parameters:
                        kwargs[k] = v
                # If no kwargs matched, skip trying this variant for this fn
                if not kwargs and len(sig.parameters)==0:
                    # try call with positional prompt if function takes 1 param
                    try:
                        if len(sig.parameters)==1:
                            # find first param name and pass prompt
                            first_param = next(iter(sig.parameters))
                            kwargs = {first_param: prompt}
                    except Exception:
                        pass
                if not kwargs and len(sig.parameters)==0:
                    # bare call
                    pass
            else:
                kwargs = pv  # unknown signature: try broad
            # attempt call
            try:
                res = fn(**kwargs) if kwargs else fn(prompt)
                # normalize result to string if possible
                if isinstance(res, dict):
                    # prefer common keys
                    for k in ("candidates","output","response","text","content"):
                        if k in res:
                            if k=="candidates" and isinstance(res[k], list) and res[k]:
                                candidate = res[k][0]
                                # try nested keys
                                if isinstance(candidate, dict):
                                    for ck in ("output","content","text"):
                                        if ck in candidate and isinstance(candidate[ck], str):
                                            return {"ok": True, "method": name, "result": candidate[ck], "attempts": attempts, "last_exception": None}
                                else:
                                    return {"ok": True, "method": name, "result": str(candidate), "attempts": attempts, "last_exception": None}
                            elif isinstance(res[k], str):
                                return {"ok": True, "method": name, "result": res[k], "attempts": attempts, "last_exception": None}
                    # fallback: return json string
                    return {"ok": True, "method": name, "result": json.dumps(res), "attempts": attempts, "last_exception": None}
                else:
                    # object with attributes
                    # try common attrs
                    text = None
                    if hasattr(res, "text"):
                        text = getattr(res, "text")
                    elif hasattr(res, "content"):
                        text = getattr(res, "content")
                    elif hasattr(res, "output"):
                        text = getattr(res, "output")
                    elif isinstance(res, str):
                        text = res
                    if text is not None:
                        return {"ok": True, "method": name, "result": str(text), "attempts": attempts, "last_exception": None}
                    # else stringified fallback
                    return {"ok": True, "method": name, "result": str(res), "attempts": attempts, "last_exception": None}
            except TypeError as te:
                # parameter mismatch likely; record and continue
                attempts.append({"call": name, "kwargs_sample": list(pv.keys()), "error": f"TypeError: {te}"})
                last_exc = te
                continue
            except Exception as e:
                attempts.append({"call": name, "kwargs_sample": list(pv.keys()), "error": repr(e)})
                last_exc = e
                continue
    # nothing worked
    return {"ok": False, "method": None, "result": None, "attempts": attempts, "last_exception": repr(last_exc)}

# REST fallback with API key
def call_rest_probe(prompt: str, api_key: str, model: Optional[str]=None) -> Dict[str,Any]:
    attempts = []
    model_list = [model] if model else MODEL_PROBES
    for base in GENERATIVE_BASES:
        for m in model_list:
            url = f"{base.rstrip('/')}/{m}:generateText"
            payload = {"prompt": {"text": prompt}, "maxOutputTokens": 1024, "temperature": 0.0}
            try:
                r = requests.post(url, params={"key": api_key}, json=payload, timeout=30)
                attempts.append({"url": url, "status": r.status_code, "body": (r.text or "")[:800]})
                if r.status_code==200:
                    # extract text if possible
                    try:
                        data = r.json()
                        if "candidates" in data and data["candidates"]:
                            c = data["candidates"][0]
                            for k in ("output","content","text","displayText"):
                                if isinstance(c, dict) and k in c:
                                    return {"ok": True, "method": f"REST {base}/{m}", "result": c[k], "attempts": attempts}
                        # try data top-level keys
                        for k in ("output","response","result"):
                            if k in data and isinstance(data[k], str):
                                return {"ok": True, "method": f"REST {base}/{m}", "result": data[k], "attempts": attempts}
                        return {"ok": True, "method": f"REST {base}/{m}", "result": json.dumps(data), "attempts": attempts}
                    except Exception:
                        return {"ok": True, "method": f"REST {base}/{m}", "result": r.text, "attempts": attempts}
            except Exception as e:
                attempts.append({"url": url, "status": None, "body": f"request exception {repr(e)}"})
    return {"ok": False, "method": None, "result": None, "attempts": attempts}

# ---------------- Streamlit UI ----------------
st.set_page_config(layout="wide", page_title="Qforia robust client")
st.title("Qforia — robust client reflection + REST fallback")

with st.sidebar:
    st.header("Auth & model")
    st.write("Paste your Gemini API key. The app will try the installed client first (dynamic), then REST fallback using the key.")
    api_key = st.text_input("Gemini API Key (required)", type="password")
    model_resource = st.text_input("Model resource (optional) e.g. models/text-bison-001", value=DEFAULT_MODEL)
    st.caption("If client reflection fails, the app will attempt REST probes using your API key.")

seed_query = st.text_input("Seed query", value="how to do call forwarding")
region = st.selectbox("Region hint", ["United States","United Kingdom","India","Canada","Australia"], index=0)
num_queries = st.number_input("Number of synthetic queries", min_value=3, max_value=200, value=DEFAULT_NUM)
extra = st.text_area("Extra prompt instructions (optional)", value="")
run = st.button("Run Fan-Out (robust)")

if "last_df" not in st.session_state:
    st.session_state["last_df"] = pd.DataFrame()
if "raw" not in st.session_state:
    st.session_state["raw"] = ""
if "diagnostic" not in st.session_state:
    st.session_state["diagnostic"] = {}

if run:
    if not api_key or not api_key.strip():
        st.error("API key required.")
    else:
        prompt = make_prompt(seed_query, region, int(num_queries), extra)
        st.info("Attempting client invocation via reflection (dynamic)...")
        client_resp = call_via_client_reflection(prompt, model_resource.strip() if model_resource else DEFAULT_MODEL)
        st.write("Client reflection attempts summary:")
        st.json({"client_ok": client_resp.get("ok"), "method": client_resp.get("method"), "attempt_count": len(client_resp.get("attempts",[]))})
        if client_resp.get("ok"):
            st.success(f"Client method succeeded: {client_resp.get('method')}")
            raw = client_resp.get("result") or ""
            st.session_state["raw"] = raw
            parsed = extract_json_array(raw)
            if parsed is None:
                st.warning("Client returned text could not be parsed as JSON. Raw output:")
                st.code(raw[:4000])
            else:
                rows = []
                for i,item in enumerate(parsed):
                    rows.append({"rank":i+1,"query":item.get("query",""),"intent":item.get("intent",""),"entity":item.get("entity",""),"variation_type":item.get("variation_type",""),"rationale":item.get("rationale","")})
                st.session_state["last_df"] = pd.DataFrame(rows)
        else:
            st.warning("Client reflection did not yield a working method. Showing attempts. Will now try REST fallback using your API key.")
            st.json(client_resp.get("attempts",[])[:30])
            rest_resp = call_rest_probe(prompt, api_key.strip(), model_resource.strip() if model_resource else None)
            if rest_resp.get("ok"):
                st.success(f"REST fallback succeeded via {rest_resp.get('method')}")
                raw = rest_resp.get("result") or ""
                st.session_state["raw"] = raw
                parsed = extract_json_array(raw)
                if parsed is None:
                    st.warning("REST returned text could not be parsed as JSON. Raw output:")
                    st.code(raw[:4000])
                else:
                    rows = []
                    for i,item in enumerate(parsed):
                        rows.append({"rank":i+1,"query":item.get("query",""),"intent":item.get("intent",""),"entity":item.get("entity",""),"variation_type":item.get("variation_type",""),"rationale":item.get("rationale","")})
                    st.session_state["last_df"] = pd.DataFrame(rows)
            else:
                st.error("REST fallback also failed. See attempt summaries below.")
                st.subheader("Client attempts (sample)")
                st.dataframe(pd.DataFrame(client_resp.get("attempts",[])[:40]))
                st.subheader("REST attempts")
                st.dataframe(pd.DataFrame(rest_resp.get("attempts",[])[:40]))
                st.code(f"Client last_exception: {client_resp.get('last_exception')}\nREST last attempts shown above.")

# Display results if present
if not st.session_state["last_df"].empty:
    st.subheader("Synthetic Queries")
    st.dataframe(st.session_state["last_df"])
    st.download_button("Export CSV", data=st.session_state["last_df"].to_csv(index=False).encode("utf-8"), file_name="qforia_results.csv", mime="text/csv")
    st.download_button("Export JSON", data=st.session_state["last_df"].to_json(orient="records", force_ascii=False), file_name="qforia_results.json", mime="application/json")

if st.session_state.get("raw"):
    st.markdown("---")
    st.subheader("Raw generation output (truncated)")
    st.code(st.session_state["raw"][:4000])

st.caption("This app tries many client call shapes dynamically, then falls back to REST. If both fail, copy the attempt tables and paste them here so I can interpret the exact error codes and shape messages and recommend the precise next step.")
