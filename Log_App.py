# app.py
"""
Qforia robust — now drives session objects (ChatSession) when reflection returns one,
then falls back to REST probes. Minimal UI: paste Gemini API key & optional model resource.
Install: pip install streamlit pandas requests
If google-generativeai exists in the environment, the app will try to use it dynamically.
"""

import os, json, re, inspect, time
from typing import Optional, List, Dict, Any
import requests
import streamlit as st
import pandas as pd

# Try to import installed client (may or may not be present)
try:
    import google.generativeai as genai  # type: ignore
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

# Config / probes
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

# ---------- helpers ----------
def make_prompt(seed: str, region: str, n: int, extra: str="") -> str:
    ts = pd.Timestamp.utcnow().isoformat() + "Z"
    return f"""You are an assistant that MUST OUTPUT valid JSON only: a JSON array of exactly {n} objects.
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

def _normalize_to_text(res: Any) -> Optional[str]:
    """Try to pull text from dict/object/str fallback."""
    if res is None:
        return None
    if isinstance(res, str):
        return res
    if isinstance(res, dict):
        # common keys
        for k in ("output", "content", "text", "message", "displayText", "result"):
            v = res.get(k)
            if isinstance(v, str):
                return v
        # candidates
        if "candidates" in res and isinstance(res["candidates"], list) and res["candidates"]:
            cand = res["candidates"][0]
            if isinstance(cand, dict):
                for k in ("output","content","text"):
                    if k in cand and isinstance(cand[k], str):
                        return cand[k]
                return json.dumps(cand)
            return str(cand)
        return json.dumps(res)
    # object with attributes
    for attr in ("text","content","output","message","response"):
        if hasattr(res, attr):
            try:
                v = getattr(res, attr)
                if isinstance(v, str):
                    return v
            except Exception:
                pass
    # fallback to str
    try:
        return str(res)
    except Exception:
        return None

# ---------- client reflection and session driving ----------
def call_via_client_reflection_and_drive(prompt: str, model: str, max_tokens:int=1024, temperature:float=0.0) -> Dict[str,Any]:
    """
    Attempt to invoke installed client *and* if it returns a session object, try instance methods that generate text.
    Returns dict with keys: ok(bool), method(str or None), result(str or None), attempts(list), diagnostics(dict)
    """
    attempts = []
    diagnostics = {}
    if not GENAI_AVAILABLE:
        diagnostics["client_present"] = False
        return {"ok": False, "method": None, "result": None, "attempts": attempts, "diagnostics": diagnostics}

    diagnostics["client_present"] = True
    # gather candidate callables (top-level and genai.chat)
    candidate_names = []
    for name in dir(genai):
        if name.startswith("_"):
            continue
        lname = name.lower()
        if any(tok in lname for tok in ("generate","create","chat","text","model","completion","session","response","predict")):
            candidate_names.append(name)
    # include genai.chat.* if exists
    if hasattr(genai, "chat"):
        for name in dir(getattr(genai,"chat")):
            if name.startswith("_"):
                continue
            lname = name.lower()
            if any(tok in lname for tok in ("create","generate","session","send","start","response")):
                candidate_names.append(f"chat.{name}")
    # dedupe preserving order
    seen=set(); ordered=[]
    for n in candidate_names:
        if n not in seen:
            seen.add(n); ordered.append(n)

    diagnostics["candidate_names"] = ordered[:80]

    # try each callable
    for name in ordered:
        try:
            # resolve object
            if "." in name:
                parent, attr = name.split(".",1)
                obj = getattr(getattr(genai, parent), attr)
            else:
                obj = getattr(genai, name)
        except Exception as e:
            attempts.append({"name": name, "error": f"resolve_error:{repr(e)}"})
            continue
        if not callable(obj):
            attempts.append({"name": name, "note":"not_callable", "repr": repr(obj)})
            continue

        # prepare parameter variations to try
        param_variants = [
            {"model": model, "prompt": prompt, "max_output_tokens": max_tokens, "temperature": temperature},
            {"model": model, "input": prompt, "max_output_tokens": max_tokens, "temperature": temperature},
            {"model": model, "prompt": {"text": prompt}, "max_output_tokens": max_tokens, "temperature": temperature},
            {"model": model, "messages": [{"role":"user","content":prompt}], "max_output_tokens": max_tokens, "temperature": temperature},
            {"model": model, "messages": [{"role":"user","content":prompt}], "maxTokens": max_tokens, "temperature": temperature},
            {"model": model, "text": prompt, "max_output_tokens": max_tokens, "temperature": temperature},
        ]
        # try calling the function with each variant
        for pv in param_variants:
            try:
                # attempt call; prefer kwargs only if signature supports param names
                result = None
                try:
                    sig = inspect.signature(obj)
                    # keep only args that appear in signature
                    kwargs = {}
                    for k,v in pv.items():
                        if k in sig.parameters:
                            kwargs[k]=v
                    if kwargs:
                        result = obj(**kwargs)
                    else:
                        # if signature accepts positional, try single positional prompt
                        if len(sig.parameters) == 1:
                            result = obj(prompt)
                        else:
                            # try broad call with pv (may raise TypeError)
                            result = obj(**pv)
                except (ValueError, TypeError):
                    # signature introspection failed or mismatch; try direct call with pv
                    try:
                        result = obj(**pv)
                    except Exception as e:
                        # try single positional
                        try:
                            result = obj(prompt)
                        except Exception as ee:
                            raise e
                # If result is immediate text, normalize and return
                text = _normalize_to_text(result)
                if text:
                    attempts.append({"name": name, "variant": list(pv.keys()), "result_preview": text[:300]})
                    return {"ok": True, "method": name, "result": text, "attempts": attempts, "diagnostics": diagnostics}
                # If result is an object (likely a session), attempt to drive it
                attempts.append({"name": name, "variant": list(pv.keys()), "result_type": type(result).__name__})
                # inspect instance for possible generation methods
                inst = result
                inst_methods = [m for m in dir(inst) if not m.startswith("_")]
                diagnostics.setdefault("instance_methods", {})[name] = inst_methods[:200]
                # prioritized instance method names to try
                candidate_inst_methods = ["send","append","create","generate","respond","get_response","get_reply","next","stream","chat","predict","complete","submit","append_user_message"]
                # intersect with actual methods, preserving candidate order
                to_try = []
                for m in candidate_inst_methods:
                    if m in inst_methods:
                        to_try.append(m)
                # also add other callable attrs found (first 20)
                for m in inst_methods:
                    if callable(getattr(inst, m)) and m not in to_try:
                        to_try.append(m)
                # try calling instance methods with a few signatures
                inst_attempts=[]
                for meth in to_try:
                    fn = getattr(inst, meth)
                    if not callable(fn):
                        continue
                    # candidate argument shapes
                    inst_variants = [
                        {"content": prompt},
                        {"message": {"role":"user","content":prompt}},
                        {"messages": [{"role":"user","content":prompt}]},
                        {"text": prompt},
                        {"prompt": prompt},
                    ]
                    for iv in inst_variants:
                        try:
                            # prefer kwargs matching signature
                            try:
                                sig2 = inspect.signature(fn)
                                kwargs2 = {}
                                for k,v in iv.items():
                                    if k in sig2.parameters:
                                        kwargs2[k]=v
                                if kwargs2:
                                    res2 = fn(**kwargs2)
                                else:
                                    # try single positional if one param
                                    if len(sig2.parameters)==1:
                                        res2 = fn(prompt)
                                    else:
                                        # attempt with iv directly
                                        try:
                                            res2 = fn(**iv)
                                        except Exception:
                                            res2 = fn(prompt)
                            except Exception:
                                # best-effort
                                try:
                                    res2 = fn(**iv)
                                except Exception:
                                    res2 = fn(prompt)
                            text2 = _normalize_to_text(res2)
                            inst_attempts.append({"method": meth, "variant": list(iv.keys()), "result_preview": (text2 or "")[:300], "ok": bool(text2)})
                            if text2:
                                # success
                                attempts.append({"name": name, "used_instance_method": meth, "variant": list(iv.keys()), "preview": text2[:300]})
                                return {"ok": True, "method": f"{name}.{meth}", "result": text2, "attempts": attempts, "diagnostics": diagnostics}
                        except Exception as e:
                            inst_attempts.append({"method": meth, "variant": list(iv.keys()), "error": repr(e)})
                            continue
                # if we tried instance methods but none returned string, record them
                attempts.append({"name": name, "instance_attempts": inst_attempts})
            except Exception as e:
                attempts.append({"name": name, "variant": list(pv.keys()), "error": repr(e)})
                continue
    # nothing worked
    diagnostics["note"] = "no working client call found"
    return {"ok": False, "method": None, "result": None, "attempts": attempts, "diagnostics": diagnostics}

# ---------- REST fallback ----------
def call_rest_probe(prompt: str, api_key: str, model: Optional[str]=None) -> Dict[str,Any]:
    attempts=[]
    model_list = [model] if model else MODEL_PROBES
    for base in GENERATIVE_BASES:
        for m in model_list:
            url = f"{base.rstrip('/')}/{m}:generateText"
            payload = {"prompt": {"text": prompt}, "maxOutputTokens": 1024, "temperature": 0.0}
            try:
                r = requests.post(url, params={"key": api_key}, json=payload, timeout=30)
                attempts.append({"url": url, "status": r.status_code, "body": (r.text or "")[:800]})
                if r.status_code==200:
                    try:
                        data=r.json()
                        # find text
                        if isinstance(data, dict):
                            if "candidates" in data and data["candidates"]:
                                c=data["candidates"][0]
                                for k in ("output","content","text","displayText"):
                                    if isinstance(c, dict) and k in c and isinstance(c[k], str):
                                        return {"ok": True, "method": f"REST {base}/{m}", "result": c[k], "attempts": attempts}
                            for k in ("output","response","result","content"):
                                if k in data and isinstance(data[k], str):
                                    return {"ok": True, "method": f"REST {base}/{m}", "result": data[k], "attempts": attempts}
                        return {"ok": True, "method": f"REST {base}/{m}", "result": r.text, "attempts": attempts}
                    except Exception:
                        return {"ok": True, "method": f"REST {base}/{m}", "result": r.text, "attempts": attempts}
            except Exception as e:
                attempts.append({"url": url, "status": None, "body": f"request exception: {repr(e)}"})
    return {"ok": False, "method": None, "result": None, "attempts": attempts}

# ---------- Streamlit UI ----------
st.set_page_config(layout="wide", page_title="Qforia session-driven")
st.title("Qforia — session-driven client reflection + REST fallback")

with st.sidebar:
    st.header("Auth & model")
    st.write("Paste your Gemini API key. The app will try to use the installed client dynamically and if that fails, fall back to REST probes.")
    api_key = st.text_input("Gemini API Key (required)", type="password")
    model_resource = st.text_input("Model resource (optional) e.g. models/text-bison-001", value=DEFAULT_MODEL)
    st.caption("If client reflection returns a session object, the app will attempt likely instance methods to get a response.")

seed_query = st.text_input("Seed query", value="how to do call forwarding")
region = st.selectbox("Region hint", ["United States","United Kingdom","India","Canada","Australia"], index=0)
num_queries = st.number_input("Number of synthetic queries", min_value=3, max_value=200, value=DEFAULT_NUM)
extra = st.text_area("Extra prompt instructions (optional)", value="")
run = st.button("Run (session-driven)")

if "last_df" not in st.session_state:
    st.session_state["last_df"] = pd.DataFrame()
if "raw" not in st.session_state:
    st.session_state["raw"] = ""
if "diag" not in st.session_state:
    st.session_state["diag"] = {}

if run:
    if not api_key or not api_key.strip():
        st.error("API key required.")
    else:
        prompt = make_prompt(seed_query, region, int(num_queries), extra)
        st.info("Attempting dynamic client invocation and instance driving...")
        client_resp = call_via_client_reflection_and_drive(prompt, model_resource.strip() if model_resource else DEFAULT_MODEL)
        st.json({"client_ok": client_resp.get("ok"), "method": client_resp.get("method")})
        if client_resp.get("ok"):
            st.success(f"Client method succeeded: {client_resp.get('method')}")
            raw = client_resp.get("result") or ""
            st.session_state["raw"] = raw
            parsed = extract_json_array(raw)
            if parsed is None:
                st.warning("Client returned text could not be parsed as JSON. Raw output:")
                st.code(raw[:4000])
            else:
                rows=[]
                for i,it in enumerate(parsed):
                    rows.append({"rank":i+1,"query":it.get("query",""),"intent":it.get("intent",""),"entity":it.get("entity",""),"variation_type":it.get("variation_type",""),"rationale":it.get("rationale","")})
                st.session_state["last_df"] = pd.DataFrame(rows)
        else:
            st.warning("Client reflection failed to produce generation. Showing diagnostics and trying REST fallback.")
            st.subheader("Client diagnostics (partial)")
            st.json(client_resp.get("diagnostics", {}))
            st.subheader("Client attempts (sample)")
            st.dataframe(pd.DataFrame(client_resp.get("attempts", [])[:50]))
            st.info("Now attempting REST probes using your API key...")
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
                    rows=[]
                    for i,it in enumerate(parsed):
                        rows.append({"rank":i+1,"query":it.get("query",""),"intent":it.get("intent",""),"entity":it.get("entity",""),"variation_type":it.get("variation_type",""),"rationale":it.get("rationale","")})
                    st.session_state["last_df"] = pd.DataFrame(rows)
            else:
                st.error("REST fallback failed as well. See attempts above.")
                st.subheader("REST attempts")
                st.dataframe(pd.DataFrame(rest_resp.get("attempts", [])[:40]))
                st.code(f"Client diagnostics: {json.dumps(client_resp.get('diagnostics',{}), indent=2)[:3000]}")

# show results
if not st.session_state["last_df"].empty:
    st.subheader("Synthetic Queries")
    st.dataframe(st.session_state["last_df"])
    st.download_button("Export CSV", data=st.session_state["last_df"].to_csv(index=False).encode("utf-8"), file_name="qforia_results.csv", mime="text/csv")
    st.download_button("Export JSON", data=st.session_state["last_df"].to_json(orient="records", force_ascii=False), file_name="qforia_results.json", mime="application/json")

if st.session_state.get("raw"):
    st.markdown("---")
    st.subheader("Raw generation output (truncated)")
    st.code(st.session_state["raw"][:4000])

st.caption("If both client session-driving and REST probes fail, copy the 'Client diagnostics' JSON and the first REST attempts rows and paste them here; I will interpret the exact error codes and next step.")
