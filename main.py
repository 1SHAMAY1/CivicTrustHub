# ==============================================================
# CivicTrust Hub – Live Dynamic Verification Engine
#
# Features:
#   1. Real-time Live Web & News Ingestion (Zero static mock database)
#   2. DeBERTa-v3 NLI Cross-Encoder (~500 MB) for Entailment / Contradiction
#   3. Local Vector / Semantic Similarity Tensor Engine Fallback
#   4. Zero Gemini / External Closed LLM API required
#   5. 3-State Factual Verdicts: VERIFIED / CONTRADICTED / UNVERIFIED
#   6. Full Unicode Support (Hindi & Regional Indian Scripts)
# ==============================================================

from __future__ import annotations
import re
import html
import json
import math
import pathlib
import datetime
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Optional, List, Dict, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ----------------------------------------------------------------
# Live Web Ingestion (Adaptive Global & Regional News Feed)
# ----------------------------------------------------------------
def fetch_live_web_evidence(query: str, max_results: int = 8) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """
    Dynamically queries live news feeds & credible web indexes.
    Returns: (evidence_list, error_message_if_any)
    """
    # Clean query preserving Unicode words (supports Hindi, Bengali, etc.)
    clean_query = re.sub(r'[^\w\s]', ' ', query, flags=re.UNICODE).strip()
    encoded_q = urllib.parse.quote(clean_query)
    evidence: List[Dict[str, str]] = []
    network_error: Optional[str] = None

    # 1. Google News Live Search Feed
    news_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        req = urllib.request.Request(news_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=5) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            items = root.findall(".//item")
            for item in items[:max_results]:
                title = item.find("title").text if item.find("title") is not None else ""
                source_tag = item.find("source")
                source = source_tag.text if source_tag is not None else "Verified News Network"
                if title:
                    clean_title = re.sub(r' - [^-]+$', '', title).strip()
                    evidence.append({
                        "source": source,
                        "text": clean_title,
                        "origin": "Live News Feed"
                    })
    except Exception as e:
        print(f"[Web Scraper] News RSS fetch notice: {e}")
        network_error = str(e)

    # 2. Wikipedia Live Fact Index
    if len(evidence) < 3:
        try:
            wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_q}&format=json"
            req = urllib.request.Request(wiki_url, headers={"User-Agent": "CivicTrustHub/2.0"})
            with urllib.request.urlopen(req, timeout=5) as response:
                raw = json.loads(response.read().decode("utf-8"))
                for r in raw.get("query", {}).get("search", [])[:4]:
                    clean_snippet = re.sub(r'<[^<]+?>', '', r.get("snippet", ""))
                    clean_snippet = html.unescape(clean_snippet).strip()
                    if clean_snippet:
                        evidence.append({
                            "source": f"Global Registry: {r.get('title')}",
                            "text": clean_snippet,
                            "origin": "Verified Knowledge Base"
                        })
        except Exception as e:
            print(f"[Web Scraper] Wiki fetch notice: {e}")
            if not network_error:
                network_error = str(e)

    return evidence, network_error


# ----------------------------------------------------------------
# DeBERTa-v3 Natural Language Inference (NLI) Cross-Encoder Engine
# ----------------------------------------------------------------
NLI_MODEL = None
NLI_TOKENIZER = None
_NLI_TRIED = False

def get_nli_engine():
    """
    Lazy loads the DeBERTa-v3 NLI Cross-Encoder (~500MB).
    Flags _NLI_TRIED to prevent re-attempting failed downloads on every request.
    """
    global NLI_MODEL, NLI_TOKENIZER, _NLI_TRIED
    if _NLI_TRIED:
        return NLI_MODEL, NLI_TOKENIZER

    _NLI_TRIED = True
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        model_name = "cross-encoder/nli-deberta-v3-base"
        print(f"[NLI Engine] 🔄 Loading DeBERTa-v3 Cross-Encoder: {model_name}...")
        NLI_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        NLI_MODEL = AutoModelForSequenceClassification.from_pretrained(model_name)
        NLI_MODEL.eval()
        print("[NLI Engine] ✅ DeBERTa-v3 Cross-Encoder loaded successfully into memory.")
    except Exception as ex:
        print(f"[NLI Engine] ⚡ NLI Cross-Encoder running in fallback mode: {ex}")
        NLI_MODEL = None
        NLI_TOKENIZER = None

    return NLI_MODEL, NLI_TOKENIZER


def evaluate_nli_entailment(claim: str, evidence_texts: List[str]) -> List[Dict[str, float]]:
    model, tokenizer = get_nli_engine()
    if model is None or tokenizer is None:
        return []

    try:
        import torch
        features = tokenizer(
            evidence_texts,
            [claim] * len(evidence_texts),
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            scores = model(**features).logits
            probs = torch.softmax(scores, dim=1).tolist()

        # 0=contradiction, 1=entailment, 2=neutral
        results = []
        for p in probs:
            results.append({
                "contradiction": float(p[0]),
                "entailment": float(p[1]),
                "neutral": float(p[2])
            })
        return results
    except Exception as ex:
        print(f"[NLI Evaluation] Notice: {ex}")
        return []


# ----------------------------------------------------------------
# Local Tensor & TF-IDF Cosine Vector Fallback Engine
# ----------------------------------------------------------------
ENCODER_MODEL = None
_ENCODER_TRIED = False

def get_sentence_transformer():
    global ENCODER_MODEL, _ENCODER_TRIED
    if _ENCODER_TRIED:
        return ENCODER_MODEL

    _ENCODER_TRIED = True
    try:
        from sentence_transformers import SentenceTransformer
        ENCODER_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("[Local Tensor Engine] ✅ MiniLM Transformer Model loaded successfully.")
    except Exception as ex:
        print(f"[Local Tensor Engine] ⚡ SentenceTransformer initializing or fallback mode: {ex}")
        ENCODER_MODEL = None

    return ENCODER_MODEL


def compute_semantic_similarity(query: str, candidate_texts: List[str]) -> List[float]:
    model = get_sentence_transformer()
    if model is not None:
        try:
            import numpy as np
            query_vec = model.encode([query], normalize_embeddings=True)[0]
            cand_vecs = model.encode(candidate_texts, normalize_embeddings=True)
            similarities = cand_vecs @ query_vec
            return [float(s) for s in similarities]
        except Exception:
            pass

    # Deterministic token cosine vector calculation (Unicode-aware, stopword filtered)
    stopwords = {'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'is', 'was', 'for', 'on', 'with', 'by'}
    words_q = [w for w in re.findall(r'\b\w{2,}\b', query.lower(), flags=re.UNICODE) if w not in stopwords]
    if not words_q:
        return [0.0] * len(candidate_texts)

    sims = []
    for cand in candidate_texts:
        words_c = [w for w in re.findall(r'\b\w{2,}\b', cand.lower(), flags=re.UNICODE) if w not in stopwords]
        set_c = set(words_c)
        matches = sum(1 for w in words_q if w in set_c)
        denominator = math.sqrt(len(words_q) * max(len(set_c), 1))
        score = (matches / denominator) if denominator > 0 else 0.0
        sims.append(min(1.0, max(0.0, score)))

    return sims


# ----------------------------------------------------------------
# Fuzzy Query Normalization & Input Validation
# ----------------------------------------------------------------
def normalize_and_validate_query(raw_text: str) -> Tuple[Optional[str], Optional[dict]]:
    raw_stripped = raw_text.strip()
    clean_raw = re.sub(r'[^\w\s\?]', ' ', raw_stripped, flags=re.UNICODE).lower().strip()
    clean_no_punct = re.sub(r'[\?]+', '', clean_raw).strip()
    raw_words = [w for w in clean_no_punct.split() if w]

    common_greetings = {
        'hi', 'hello', 'hey', 'test', 'testing', 'good morning', 'good afternoon', 'good evening', 'bye', 'ok', 'okay', 'yes', 'no'
    }

    if clean_no_punct in common_greetings or len(raw_words) < 2 or len(clean_no_punct) < 5 or clean_no_punct.isdigit():
        return None, {
            "is_invalid": True,
            "verdict": "INVALID_QUERY",
            "message": "Please check rumors only.",
            "confidence": None,
            "ground_truth": {
                "source": "Input Sanity Filter",
                "text": "Query does not contain an assertive claim or rumor to corroborate."
            },
            "counter_en": "Please enter an assertive claim or rumor to check (e.g., 'Schools closed tomorrow due to flood').",
            "counter_hi": "कृपया जांच के लिए कोई विशिष्ट दावा या अफवाह दर्ज करें।"
        }

    # Catch conversational prompts, slang, and typo variants (e.g. "tell me about your slef", "how u doing")
    is_conversational = bool(
        re.search(r'(\btell me\b|\babout your\b|\babout yourself\b|\babout ur\b|\babout slef\b|\bwho are you\b|\bwho r u\b|\bwho is this\b|\bhow to use\b|\byou are good\b|\byou r good\b)', clean_no_punct) or
        re.search(r'^(how\s*(r|are|u|you)|what\s*(s|\'s|up|r|is)|who\s*(r|is|u)|sup|wbu)\b', clean_no_punct)
    )

    if is_conversational:
        return None, {
            "is_invalid": True,
            "verdict": "INVALID_QUERY",
            "message": "Please check rumors only.",
            "confidence": None,
            "ground_truth": {
                "source": "Input Sanity Filter",
                "text": "Conversational prompt detected. Please submit a news claim or viral rumor to verify."
            },
            "counter_en": "Please enter an assertive rumor or news claim to check.",
            "counter_hi": "कृपया जांच के लिए कोई विशिष्ट दावा या अफवाह दर्ज करें।"
        }

    # Only strip leading auxiliary if the sentence ends with a question mark (preserves "Will Smith")
    ends_with_q = raw_stripped.endswith('?')
    if ends_with_q:
        normalized = re.sub(r'^(is|are|was|were|did|does|will|can|has|have)\s+', '', clean_no_punct, flags=re.IGNORECASE).strip()
    else:
        normalized = clean_no_punct

    return normalized if normalized else clean_no_punct, None


# ----------------------------------------------------------------
# Real-Time Fact Verification & 3-State Verdict Engine
# ----------------------------------------------------------------
def verify_claim_against_web(claim_text: str) -> dict:
    normalized_claim, invalid_resp = normalize_and_validate_query(claim_text)
    if invalid_resp:
        return invalid_resp

    evidence_list, network_error = fetch_live_web_evidence(normalized_claim, max_results=8)

    # Differentiate true network failure from lack of evidence
    if not evidence_list:
        if network_error:
            return {
                "is_invalid": False,
                "verdict": "NETWORK_ERROR",
                "classification": "SERVICE TEMPORARILY UNAVAILABLE",
                "confidence": None,
                "ground_truth": {
                    "source": "Network Ingestion Stream",
                    "text": "Could not connect to live news or knowledge indices. Please check internet connection and retry."
                },
                "counter_en": "Network connection to verification streams timed out. Please retry in a few moments.",
                "counter_hi": "समाचार सत्यापन नेटवर्क से संपर्क नहीं हो सका। कृपया पुनः प्रयास करें।"
            }
        return {
            "is_invalid": False,
            "verdict": "UNVERIFIED",
            "classification": "UNVERIFIED CLAIM – Zero Credible News Confirmation",
            "confidence": None,
            "ground_truth": {
                "source": "Global News Sweep",
                "text": f"Extensive real-time news scans found zero verified official dispatches regarding: \"{claim_text}\"."
            },
            "counter_en": f"No verified news agencies or disaster management organizations have reported \"{claim_text}\". Exercise caution against unverified forwards.",
            "counter_hi": f"अफवाह से बचें: \"{claim_text}\" के संबंध में किसी भी आधिकारिक समाचार एजेंसी से कोई पुष्टि नहीं मिली है।"
        }

    candidate_texts = [e["text"] for e in evidence_list]

    # 1. Primary Engine: DeBERTa NLI Cross-Encoder
    nli_results = evaluate_nli_entailment(normalized_claim, candidate_texts)

    if nli_results:
        best_signal_idx = 0
        best_signal_val = 0.0
        best_verdict_type = "neutral"

        for idx, res in enumerate(nli_results):
            e_val = res["entailment"]
            c_val = res["contradiction"]
            top_val = max(e_val, c_val)
            if top_val > best_signal_val:
                best_signal_val = top_val
                best_signal_idx = idx
                best_verdict_type = "entailment" if e_val >= c_val else "contradiction"

        best_match = evidence_list[best_signal_idx]
        best_res = nli_results[best_signal_idx]

        if best_verdict_type == "entailment" and best_res["entailment"] >= 0.60:
            confidence = round(best_res["entailment"], 2)
            return {
                "is_invalid": False,
                "verdict": "VERIFIED",
                "classification": "VERIFIED EVENT – Corroborated via Cross-Attention Inference",
                "confidence": confidence,
                "ground_truth": {"source": best_match["source"], "text": best_match["text"]},
                "counter_en": f"CORROBORATED: Live records from {best_match['source']} confirm this statement: \"{best_match['text']}\".",
                "counter_hi": f"सत्यापित समाचार: {best_match['source']} के आधिकारिक रिकॉर्ड इस दावे की पुष्टि करते हैं: \"{best_match['text']}\"।"
            }
        elif best_verdict_type == "contradiction" and best_res["contradiction"] >= 0.55:
            confidence = round(best_res["contradiction"], 2)
            return {
                "is_invalid": False,
                "verdict": "CONTRADICTED",
                "classification": "CONTRADICTED / REFUTED – Disproved by Reported Dispatches",
                "confidence": confidence,
                "ground_truth": {"source": best_match["source"], "text": best_match["text"]},
                "counter_en": f"DISPROVED: Reported evidence from {best_match['source']} refutes this assertion: \"{best_match['text']}\".",
                "counter_hi": f"अपुष्ट एवं खंडित: {best_match['source']} की रिपोर्टें इस दावे का खंडन करती हैं: \"{best_match['text']}\"।"
            }

    # 2. Secondary Engine: Semantic Similarity & Polarity/Temporal Alignment
    similarities = compute_semantic_similarity(normalized_claim, candidate_texts)
    best_idx = int(max(range(len(similarities)), key=lambda i: similarities[i]))
    best_sim = round(similarities[best_idx], 3)
    best_match = evidence_list[best_idx]

    q_lower = normalized_claim.lower()
    ev_lower = best_match["text"].lower()
    current_year = datetime.datetime.now().year

    # Check negations preserving contractions
    query_neg = bool(re.search(r"\b(not|never|no|n't|unmarried|unreleased|fake|false|hoax|disproved)\b", q_lower))
    ev_neg = bool(re.search(r"\b(not|never|no|n't|unmarried|unreleased|unsubstantiated|fake|false|rumor|denied)\b", ev_lower))

    is_claiming_past = bool(re.search(r'\b(released|launched|won|died|killed|premiered|happened)\b', q_lower))
    # Exclude current year (2026) so present events aren't marked future
    future_year_pattern = rf'\b({current_year+1}|{current_year+2}|{current_year+3})\b'
    is_future_event = bool(re.search(r'\b(to be released|will be released|being released in|scheduled for|slated for|anticipated in)\b', ev_lower) or
                          re.search(future_year_pattern, ev_lower))

    # Marriage entity check (avoid Marriott false matches)
    is_marriage_claim = bool(re.search(r'\bmarri(ed|age)\b|\bwedding\b', q_lower))
    if is_marriage_claim and not query_neg:
        mentions_unmarried = bool(re.search(r'\b(unmarried|bachelor|single|never married|not married)\b', ev_lower))
        relatives_only = bool(re.search(r'\b(brother|sister|father|mother|daughter|son|wife of|husband of)\b', ev_lower))
        if mentions_unmarried or relatives_only:
            return {
                "is_invalid": False,
                "verdict": "CONTRADICTED",
                "classification": "CONTRADICTED – Unsubstantiated by Credible Profiles",
                "confidence": best_sim if best_sim > 0.4 else None,
                "ground_truth": {"source": best_match["source"], "text": best_match["text"]},
                "counter_en": f"UNSUBSTANTIATED: Official biographical records do not verify this marriage claim. Source ({best_match['source']}): \"{best_match['text']}\".",
                "counter_hi": f"अपुष्ट: आधिकारिक रिकॉर्ड इस विवाह के दावे की पुष्टि नहीं करते हैं।"
            }

    if is_claiming_past and is_future_event and not ('future' in q_lower or 'upcoming' in q_lower):
        return {
            "is_invalid": False,
            "verdict": "CONTRADICTED",
            "classification": "CONTRADICTED – Event Scheduled in Future (Not Yet Occurred)",
            "confidence": 0.90,
            "ground_truth": {"source": best_match["source"], "text": best_match["text"]},
            "counter_en": f"FUTURE EVENT: Official dispatches confirm this project is scheduled for upcoming release, but has NOT yet occurred. Source ({best_match['source']}): \"{best_match['text']}\".",
            "counter_hi": f"भविष्य की घोषणा: आधिकारिक सूचना के अनुसार यह अभी रिलीज़ नहीं हुआ है।"
        }

    # Similarity gating: only evaluate polarity if similarity threshold is met
    if best_sim >= 0.45:
        if query_neg:
            if ev_neg or (is_marriage_claim and 'wife' not in ev_lower and 'husband' not in ev_lower):
                verdict = "VERIFIED"
                classification = "VERIFIED – Corroborated with Official Negative Records"
            else:
                verdict = "CONTRADICTED"
                classification = "CONTRADICTED – Disputed by Reported Dispatches"
        else:
            if ev_neg:
                verdict = "CONTRADICTED"
                classification = "CONTRADICTED – News Sources Refute Claim"
            else:
                verdict = "VERIFIED"
                classification = "VERIFIED – Corroborated by Live Web & News Streams"
        confidence = best_sim
    else:
        verdict = "UNVERIFIED"
        classification = "UNVERIFIED – Insufficient Corroborating Dispatches"
        confidence = None

    return {
        "is_invalid": False,
        "verdict": verdict,
        "classification": classification,
        "confidence": confidence,
        "ground_truth": {"source": best_match["source"], "text": best_match["text"]},
        "counter_en": f"Dispatch from {best_match['source']}: \"{best_match['text']}\".",
        "counter_hi": f"रिपोर्ट: \"{best_match['text']}\"।"
    }


# ----------------------------------------------------------------
# FastAPI Web Application (Secure Setup)
# ----------------------------------------------------------------
app = FastAPI(title="CivicTrust Hub Real-Time Verification API", version="2.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = pathlib.Path(__file__).parent

class RumorVerifyRequest(BaseModel):
    text: str
    platform: Optional[str] = "Live Feed"

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "engine": "DeBERTa-v3 NLI Cross-Encoder + Dynamic Web Ingestion",
        "nli_model": "cross-encoder/nli-deberta-v3-base",
        "verdict_model": "3-State (VERIFIED / CONTRADICTED / UNVERIFIED)",
        "license": "Apache-2.0 / MIT (100% Free FOSS)",
        "database_dependency": False
    }

@app.post("/api/rumors/verify")
def rumors_verify(body: RumorVerifyRequest):
    result = verify_claim_against_web(body.text)
    return JSONResponse(content=result)

@app.get("/")
def serve_index():
    index_path = ROOT_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return JSONResponse({"message": "CivicTrust Hub API active. index.html not found."}, status_code=404)
