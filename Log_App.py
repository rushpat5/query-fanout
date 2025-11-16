# app.py
"""
Qforia-approx Streamlit app (Gemini REST) — attempts to approach Qforia-level fidelity.

Features:
 - Accepts Gemini API key + model resource name for REST generateText calls.
 - Optional upload of ground-truth CSV (Search Console style) to build few-shot examples and ranking signals.
 - Optional upload of local-entities CSV (columns: entity, type, region optional) to anchor outputs.
 - Optional OpenAI API key for embeddings (better semantic matching). If not provided, uses fuzzy matching.
 - Builds region-aware few-shot prompts, generates synthetic queries, scores and ranks them using available signals.
 - Exports CSV/JSON.

Notes:
 - You MUST provide a working model resource your key can access, e.g. "models/text-bison-001" or "models/gemini-1.5-pro".
 - This app is a best-effort implementation. Production fidelity requires true query logs + fine-tuning which are out of scope.
"""

import os
import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple
import math
import difflib

import requests
import streamlit as st
import pandas as pd
import numpy as np

# Optional OpenAI for embeddings (if you supply an OpenAI key)
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    openai = None
    OPENAI_AVAILABLE = False

# ---------------- CONFIG
GENERATIVE_BASE = "https://generativelanguage.googleapis.com/v1"  # using v1; change if needed
DEFAULT_NUM_QUERIES = 40
DEFAULT_MODEL_RESOURCE = os.getenv("GEMINI_MODEL", "models/text-bison-001")
PROBE_MODEL_CANDIDATES = [
    "models/gemini-1.5-pro","models/gemini-1.5","models/gemini-1.0",
    "models/gemini-pro","models/text-bison-001","models/chat-bison-001"
]

REGIONS = [
    {"name":"United States","code":"US","language":"en-US"},
    {"name":"United Kingdom","code":"GB","language":"en-GB"},
    {"name":"India","code":"IN","language":"en-IN"},
    {"name":"Canada","code":"CA","language":"en-CA"},
    {"name":"Australia","code":"AU","language":"en-AU"},
]

# ---------------- Helpers
def make_prompt(seed_query: str, surface: str, region: Dict[str,str], num_queries: int, few_shot_examples: List[Dict[str,str]], anchors: List[str], extra_instructions: str="") -> str:
    """
    Build a robust, few-shot, region-aware prompt that requests strict JSON.
    few_shot_examples: list of dicts {"seed":..., "examples":[...sub queries...] } - will be shown as example pairs.
    anchors: list of local brand/entity anchors to bias generation.
    """
    ts = pd.Timestamp.utcnow().isoformat() + "Z"
    region_hint = f"{region.get('name')} (country_code={region.get('code')}, language={region.get('language')})"
    # Build few-shot block
    few_shot_block = ""
    for ex in few_shot_examples:
        sq = ex.get("seed")
        subqs = ex.get("examples", [])[:6]
        few_shot_block += f"\nSeed: {sq}\nFan-out examples: {', '.join(subqs)}\n"
    anchors_text = ""
    if anchors:
        anchors_text = "Use these local anchors when appropriate: " + ", ".join(anchors) + "."
    prompt = f"""
You are an assistant that generates a list of synthetic, user-style search queries derived from a seed query.
Produce exactly {num_queries} items in JSON array format. Each item must be an object with fields:
  - query (string),
  - intent (string): one of informational, commercial, transactional, navigational, investigational,
  - entity (string): the main entity if present, else empty string,
  - variation_type (string): paraphrase, narrowing, expansion, entity-focus, question-form, long-tail, comparative,
  - rationale (string): one-sentence reason.

Seed query: "{seed_query}"
Target surface: {surface}
Region: {region_hint}
Timestamp: {ts}

Requirements:
 - Output EXACTLY a JSON array and NOTHING else (no commentary, no markdown).
 - Bias phrasing to the provided locale (spelling, local brands, place names, currencies).
 - Use the few-shot examples below as a template for style and variety:{few_shot_block}

Anchors / Local entities: {anchors_text}

Additional instructions:
{extra_instructions}

Produce the JSON now.
"""
    return prompt.strip()

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

# REST call (API key required)
def call_generative_rest_api(prompt: str, api_key: str, model_resource: str, max_output_tokens: int=1024, temperature: float=0.0, timeout_s:int=60) -> Tuple[bool,int,str]:
    """
    Returns (ok:Boolean, status:int, body:str)
    model_resource should be like 'models/text-bison-001' or 'models/gemini-1.5-pro'
    """
    url = f"{GENERATIVE_BASE.rstrip('/')}/{model_resource}:generateText"
    params = {"key": api_key}
    payload = {"prompt": {"text": prompt}, "temperature": temperature, "maxOutputTokens": int(max_output_tokens)}
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(url, params=params, json=payload, headers=headers, timeout=timeout_s)
    except Exception as e:
        return False, None, f"request exception: {e}"
    status = resp.status_code
    body = resp.text or ""
    if status != 200:
        return False, status, body
    # try parse to extract candidate text snippets
    try:
        data = resp.json()
        if isinstance(data, dict):
            if "candidates" in data and isinstance(data["candidates"], list) and data["candidates"]:
                cand = data["candidates"][0]
                for k in ("output","content","text","displayText"):
                    if isinstance(cand, dict) and k in cand and isinstance(cand[k], str):
                        return True, status, cand[k]
            for k in ("output","response","result","content"):
                if k in data and isinstance(data[k], str):
                    return True, status, data[k]
        # fallback: pretty JSON
        return True, status, json.dumps(data, indent=2)
    except Exception:
        return True, status, body

# Optional embeddings via OpenAI (if provided)
def get_openai_embeddings_bulk(texts: List[str], openai_key: str, model:str="text-embedding-3-small") -> List[List[float]]:
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai package not installed.")
    openai.api_key = openai_key
    out = []
    batch = 50
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        resp = openai.Embedding.create(input=chunk, model=model)
        for item in resp["data"]:
            out.append(item["embedding"])
    return out

def cosine_sim(a: List[float], b: List[float]) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.linalg.norm(a)==0 or np.linalg.norm(b)==0:
        return 0.0
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

# Simple heuristic intent classifier (fallback)
def heuristic_intent(query: str) -> str:
    q = query.lower()
    if any(tok in q for tok in ["buy","price","cost","coupon","order","best way","best price"]):
        return "commercial"
    if any(tok in q for tok in ["how to","how do i","how can i","how to set","setup","install","enable","disable","troubleshoot","fix"]):
        return "informational"
    if any(tok in q for tok in ["near me","location","hours","address","closest","where"]):
        return "navigational"
    if any(tok in q for tok in ["compare","vs","difference","vs."]):
        return "investigational"
    return "informational"

# Ranking / scoring function combining signals
def score_candidate(query: str, entity: str, variation_type: str, rationale: str,
                    ground_truth_df: Optional[pd.DataFrame], gt_lookup:Dict[str,float],
                    embeddings_candidates: Optional[List[List[float]]], query_index:int,
                    passage_embs: Optional[List[float]] = None) -> float:
    """
    Returns a composite score (0..1). Signals:
      - presence in ground-truth (normalized clicks/impressions)
      - semantic similarity to ground-truth if embeddings available
      - simple heuristics (entity presence boosts)
    """
    score = 0.0
    # ground-truth direct lookup (exact match)
    if gt_lookup:
        freq = gt_lookup.get(query.lower(), 0.0)
        score += 0.5 * min(1.0, freq)  # freq should be normalized earlier
    # embedding similarity to top ground-truth example if available
    if embeddings_candidates and passage_embs is not None:
        try:
            sim = cosine_sim(passage_embs, embeddings_candidates[query_index])
            score += 0.3 * sim
        except Exception:
            pass
    # entity boost
    if entity:
        score += 0.1
    # variation type heuristic
    if variation_type == "question-form" or variation_type=="how_to":
        score += 0.05
    return min(1.0, score)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Qforia-approx (Gemini)", layout="wide")
st.title("Qforia-approx — Query Fan-Out (aim: high fidelity)")

with st.sidebar:
    st.header("Authentication & Inputs")
    st.write("Provide your Gemini API key + model resource name (required). Optional: provide OpenAI API key for embeddings, upload ground-truth CSV and local-entities CSV.")
    gem_key = st.text_input("Gemini API Key (required)", type="password")
    model_resource = st.text_input("Model resource (required) e.g. models/text-bison-001", value=DEFAULT_MODEL_RESOURCE)
    openai_key = st.text_input("OpenAI API Key (optional, for embeddings)", type="password")

    st.markdown("---")
    st.subheader("Optional data")
    st.write("Upload ground-truth CSV (columns: query, clicks, impressions, country optional).")
    gt_file = st.file_uploader("Ground-truth CSV", type=["csv"])
    st.write("Upload local-entities CSV (columns: entity,type,region optional).")
    entities_file = st.file_uploader("Entities CSV (optional)", type=["csv"])

    st.markdown("---")
    st.write("Generation controls")
    num_queries = st.number_input("Target number of synthetic queries", min_value=5, max_value=200, value=DEFAULT_NUM_QUERIES)
    surface = st.selectbox("Target surface", ["AI Overview", "AI Mode"], index=0)
    st.write("Prompt tuning (optional)")
    extra_instructions = st.text_area("Extra prompt instructions", value="")

# Main form
col1, col2 = st.columns([3,1])
with col1:
    seed_query = st.text_input("Seed query", value="how to do call forwarding")
    region_choice = st.selectbox("Region", [r["name"] for r in REGIONS], index=0)
    region = next((r for r in REGIONS if r["name"]==region_choice), REGIONS[0])
    gen_btn = st.button("Run Fan-Out (generate)")

with col2:
    st.write("Exports")
    export_csv = st.button("Export last results to CSV")
    export_json = st.button("Export last results to JSON")

# Load optional files
ground_truth_df = None
gt_lookup = {}
if gt_file is not None:
    try:
        ground_truth_df = pd.read_csv(gt_file)
        # normalize queries column
        if "query" not in ground_truth_df.columns:
            st.error("Ground-truth CSV must contain a 'query' column.")
            ground_truth_df = None
        else:
            ground_truth_df['query_norm'] = ground_truth_df['query'].astype(str).str.strip().str.lower()
            # produce a normalized frequency score (simple: clicks normalized)
            if 'clicks' in ground_truth_df.columns:
                total = ground_truth_df['clicks'].sum()
                if total > 0:
                    ground_truth_df['freq_norm'] = ground_truth_df['clicks'] / total
                else:
                    ground_truth_df['freq_norm'] = 0.0
            else:
                # fallback: uniform
                ground_truth_df['freq_norm'] = 1.0 / max(1, len(ground_truth_df))
            gt_lookup = dict(zip(ground_truth_df['query_norm'], ground_truth_df['freq_norm']))
    except Exception as e:
        st.error(f"Failed to load ground-truth CSV: {e}")
        ground_truth_df = None

anchors = []
if entities_file is not None:
    try:
        ent_df = pd.read_csv(entities_file)
        if 'entity' in ent_df.columns:
            anchors = list(ent_df['entity'].astype(str).dropna().unique())[:100]
        else:
            st.warning("Entities CSV should include an 'entity' column; ignoring file.")
    except Exception as e:
        st.warning(f"Failed to parse entities file: {e}")

# Session storage
if 'last_results' not in st.session_state:
    st.session_state['last_results'] = pd.DataFrame()
if 'raw_generation' not in st.session_state:
    st.session_state['raw_generation'] = ""

# Generate
if gen_btn:
    # quick validation
    if not gem_key or not gem_key.strip():
        st.error("Gemini API key is required.")
    elif not model_resource or not model_resource.strip():
        st.error("Model resource is required (e.g., models/text-bison-001).")
    else:
        prompt_examples = []
        # build few-shot examples from top ground-truth if present
        if ground_truth_df is not None:
            # pick top unique seeds: for simplicity take top 5 queries as seeds with their variants
            top = ground_truth_df.nlargest(10, 'freq_norm', keep='all') if 'freq_norm' in ground_truth_df.columns else ground_truth_df.head(10)
            for i, row in top.head(5).iterrows():
                seed = row['query']
                # create naive sub-queries by simple paraphrase heuristics (split terms etc.) as examples
                subs = [seed, f"how to {seed}", f"{seed} on iPhone", f"{seed} for business"]
                prompt_examples.append({"seed": seed, "examples": subs})
        else:
            # generic few-shot examples
            prompt_examples = [
                {"seed":"how to reset iphone","examples":["how to reset iphone 12","how to factory reset iphone","erase iphone data"]},
                {"seed":"best running shoes","examples":["best running shoes for flat feet","top running shoes 2024","budget running shoes review"]}
            ]

        prompt = make_prompt(seed_query, surface, region, int(num_queries), prompt_examples, anchors[:20], extra_instructions)

        # Call Gemini REST
        with st.spinner("Calling Gemini (REST) — this may take a few seconds..."):
            ok, status, body = call_generative_rest_api(prompt, gem_key.strip(), model_resource.strip(), max_output_tokens=1200, temperature=0.0)
        st.session_state['raw_generation'] = body
        if not ok:
            st.error(f"Generation failed: HTTP {status} — {body[:1000]}")
        else:
            # parse and normalize
            parsed = extract_json_array(body)
            if parsed is None:
                st.warning("Generation returned non-JSON or parsing failed. Showing raw output for debugging.")
                st.code(body[:4000])
            else:
                # normalize parsed items
                items = []
                for i, it in enumerate(parsed):
                    q = it.get('query','').strip()
                    if not q:
                        continue
                    intent = it.get('intent') or heuristic_intent(q)
                    entity = it.get('entity','') or ''
                    variation = it.get('variation_type','')
                    rationale = it.get('rationale','')
                    items.append({"rank_gen": i+1, "query": q, "intent": intent, "entity": entity, "variation_type": variation, "rationale": rationale})
                df_gen = pd.DataFrame(items)

                # embeddings optional
                embeddings_candidates = None
                passage_embs = None
                if openai_key and openai_key.strip():
                    if not OPENAI_AVAILABLE:
                        st.warning("OpenAI package not installed; embeddings skipped.")
                    else:
                        try:
                            queries = df_gen['query'].tolist()
                            emb_vecs = get_openai_embeddings_bulk(queries, openai_key.strip())
                            embeddings_candidates = emb_vecs
                        except Exception as e:
                            st.warning(f"Embeddings failed: {e}")
                            embeddings_candidates = None
                # compute GT-based lookup normalized
                gt_lookup_norm = gt_lookup

                # optional semantic similarity to top GT (if embeddings)
                if embeddings_candidates and ground_truth_df is not None and openai_key:
                    # compute embedding for top ground-truth query (or centroid)
                    try:
                        gt_texts = ground_truth_df['query'].astype(str).tolist()[:100]
                        gt_embs = get_openai_embeddings_bulk(gt_texts, openai_key.strip())
                        # compute centroid
                        centroid = np.mean(np.array(gt_embs), axis=0).tolist()
                        passage_embs = centroid
                    except Exception:
                        passage_embs = None

                # scoring
                scored = []
                for idx, row in df_gen.iterrows():
                    qtext = row['query']
                    # exact match GT freq
                    freq = gt_lookup_norm.get(qtext.lower(), 0.0) if gt_lookup_norm else 0.0
                    # embedding-based similarity if available (use centroid)
                    emb_sim = 0.0
                    if embeddings_candidates and passage_embs:
                        try:
                            emb_sim = cosine_sim(passage_embs, embeddings_candidates[idx])
                        except Exception:
                            emb_sim = 0.0
                    # heuristic boosts
                    ent_boost = 0.05 if row['entity'] else 0.0
                    # compute composite score
                    score = 0.6*freq + 0.3*emb_sim + ent_boost
                    # fallback normalize
                    score = float(min(1.0, score))
                    scored.append({**row.to_dict(), "gt_freq": freq, "emb_sim": emb_sim, "score": score})
                res_df = pd.DataFrame(scored).sort_values("score", ascending=False).reset_index(drop=True)
                st.session_state['last_results'] = res_df

# Display results
if not st.session_state['last_results'].empty:
    st.subheader("Generated Candidates (ranked)")
    df = st.session_state['last_results']
    st.dataframe(df)
    if export_csv:
        st.download_button("Download CSV", data=df.to_csv(index=False).encode('utf-8'), file_name="qforia_approx_results.csv", mime="text/csv")
    if export_json:
        st.download_button("Download JSON", data=df.to_json(orient="records", force_ascii=False), file_name="qforia_approx_results.json", mime="application/json")

    st.markdown("---")
    st.subheader("Utilities / Validation")
    st.write("1) Inspect raw generation output (for debugging)")
    if st.session_state.get('raw_generation'):
        st.code(st.session_state['raw_generation'][:4000])
    st.write("2) Quick overlap metrics with provided ground-truth (if available)")
    if ground_truth_df is not None:
        gen_set = set(df['query'].str.lower().tolist())
        gt_set = set(ground_truth_df['query_norm'].tolist())
        overlap = len(gen_set & gt_set)
        st.write(f"Exact overlap with GT queries: {overlap} / {len(gen_set)}")
        # show sample matches
        overlap_samples = list(gen_set & gt_set)[:10]
        if overlap_samples:
            st.write("Overlap samples:", overlap_samples)

st.markdown("---")
st.caption("This app uses generation + few-shot + anchors + ground-truth ranking to approximate production fidelity. For best results: supply ground-truth query logs, add many local anchors, and iterate prompt & example selections. Exact parity with proprietary systems requires internal telemetry and possibly fine-tuning.")
