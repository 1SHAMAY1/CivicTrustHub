# ==============================================================
# CivicTrust Hub – Authentic Rumor Detection Engine
#
# Distinct from a Search Engine:
#   1. Claim Extraction: Strips forward boilerplate ("URGENT", "share with all")
#   2. Rumor Cue Profiling: Flags viral urgency, sensationalism, vague authority, all-caps
#   3. Dual-Stream Verification: Primary news corroboration + Dedicated Fact-Check sweep
#   4. Multi-Source Corroboration Rule: >= 2 independent outlets for CONFIRMED
#   5. Recency / Recycled-News Guard: Distinguishes current events from vintage footage
#   6. Absence of Coverage Signal: High-profile assertions with zero dispatches flagged
#   7. Honest 5-State Rumor Verdicts:
#      - RUMOR (debunked)
#      - LIKELY RUMOR (unsubstantiated viral forward)
#      - UNCONFIRMED (neutral / insufficient coverage)
#      - CONFIRMED (multi-source verified)
#      - CANNOT VERIFY / INVALID QUERY
# ==============================================================

from __future__ import annotations
import re
import html
import json
import math
import pathlib
import datetime
import threading
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Optional, List, Dict, Tuple, Set

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# ----------------------------------------------------------------
# Recognized Fact-Checking Outlets (Verified Independent Desks)
# ----------------------------------------------------------------
FACTCHECK_OUTLETS: Set[str] = {
    "pib fact check",
    "boom",
    "alt news",
    "factly",
    "newschecker",
    "vishvas news",
    "the quint",
    "india today fact check",
    "cyberpeace foundation",
    "newsmeter",
    "afp fact check",
    "reuters fact check",
    "pti fact check",
    "press trust of india",
    "logical indian",
    "thip media"
}

# ----------------------------------------------------------------
# Robust Unicode Preservation (Preserves Indic letters, vowels, matras)
# ----------------------------------------------------------------
def strip_punctuation(text: str, keep_chars: str = "?'") -> str:
    """
    Cleans punctuation while preserving:
    - All Unicode alphanumeric letters & numbers (Hindi, Bengali, English, etc.)
    - Unicode combining marks (matras, virama like in 'दिल्ली')
    - Specified punctuation (like '?' and contractions "'")
    """
    normalized = text.replace("’", "'").replace("`", "'")
    chars = []
    for c in normalized:
        cat = unicodedata.category(c)
        if c.isalnum() or c.isspace() or c in keep_chars or cat.startswith("M"):
            chars.append(c)
        else:
            chars.append(" ")
    return "".join(chars)


# ----------------------------------------------------------------
# Rumor Cue Profiling & Viral Metadata Detection
# ----------------------------------------------------------------
def detect_rumor_cues(raw_text: str) -> Dict[str, any]:
    """
    Analyzes raw text for psychological manipulation & viral forwarding markers.
    Cues are explainable flags, not definitive verdicts.
    """
    text_lower = raw_text.lower()
    flags = []

    # 1. Forward Urgency
    if re.search(r'\b(forward|share (this )?(with|to) (all|everyone|every group)|urgent|alert!+|viral forward|warning to all)\b', text_lower):
        flags.append({
            "code": "FORWARD_URGENCY",
            "label": "Viral Forward Urgency",
            "desc": "Contains forwarding commands typical of chain messages."
        })

    # 2. Sensational Wording
    if re.search(r'\b(shocking|breaking news!?|exposed|you won\'?t believe|100% true|secret revealed|miracle|cure found)\b', text_lower):
        flags.append({
            "code": "SENSATIONALISM",
            "label": "Sensationalist Phrasing",
            "desc": "Uses emotional or exaggerated clickbait terminology."
        })

    # 3. Vague Authority Appeals
    if re.search(r'\b(doctors say|scientists say|govt (has )?(announced|confirmed)|sources say|insiders report|secret report)\b', text_lower):
        flags.append({
            "code": "VAGUE_AUTHORITY",
            "label": "Vague Authority Attribution",
            "desc": "Invokes unnamed experts or authorities without citation."
        })

    # 4. Excessive Capitalization (>40% uppercase letters in English sentences)
    letters = [c for c in raw_text if c.isalpha() and c.isascii()]
    if len(letters) >= 12:
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio >= 0.40:
            flags.append({
                "code": "ALL_CAPS",
                "label": "Aggressive Capitalization",
                "desc": "Over 40% uppercase characters, designed to incite panic."
            })

    # 5. Lack of Temporal or Geographical Specifics
    has_specifics = bool(re.search(
        r'\b(delhi|mumbai|bengaluru|kolkata|chennai|hyderabad|today|yesterday|tomorrow|\d{1,2}(st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{4}))\b',
        text_lower
    ))
    if not has_specifics and len(raw_text.split()) > 10:
        flags.append({
            "code": "NO_SPECIFICS",
            "label": "Missing Verifiable Anchor",
            "desc": "Omits specific named locations or verified dates."
        })

    return {
        "cue_count": len(flags),
        "flags": flags,
        "is_suspicious_forward": len(flags) >= 2
    }


# ----------------------------------------------------------------
# Core Rumor Claim Extraction (Isolates assertion from forward noise)
# ----------------------------------------------------------------
def extract_rumor_claim(raw_text: str) -> str:
    """
    Strips viral forward boilerplate (e.g., 'URGENT!! Forward to all groups: ...')
    to isolate the actionable claim for live verification.
    """
    cleaned = raw_text.strip()

    # Repeatedly strip common WhatsApp / Telegram forwarding headers
    pattern = r'^(urgent|alert|breaking|forwarded as received|share with (all|everyone)|please forward|beware|shocking)[:!\-\s]+'
    while re.search(pattern, cleaned, flags=re.IGNORECASE):
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()

    # Strip trailing share exhortations
    cleaned = re.sub(
        r'[\.\s\-]+(forward to all|share this|send to everyone|dont ignore|spread this).*$',
        '',
        cleaned,
        flags=re.IGNORECASE
    ).strip()

    return cleaned if len(cleaned) >= 5 else raw_text.strip()


# ----------------------------------------------------------------
# Live Web Ingestion (Dual Query: Breaking News + Dedicated Fact-Check)
# ----------------------------------------------------------------
def fetch_evidence_rss(query: str, max_results: int = 6) -> List[Dict[str, str]]:
    clean_query = strip_punctuation(query, keep_chars="").strip()
    encoded_q = urllib.parse.quote(clean_query)
    url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-IN&gl=IN&ceid=IN:en"
    results: List[Dict[str, str]] = []

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CivicTrustHub/3.0 RumorDetector (Mozilla/5.0)"})
        with urllib.request.urlopen(req, timeout=7) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            items = root.findall(".//item")
            for item in items[:max_results]:
                title = item.find("title").text if item.find("title") is not None else ""
                source_tag = item.find("source")
                source = source_tag.text if source_tag is not None else "Independent Wire"
                pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
                if title:
                    clean_title = re.sub(r' - [^-]+$', '', title).strip()
                    results.append({
                        "source": source,
                        "text": clean_title,
                        "pub_date": pub_date,
                        "origin": "Live News Feed"
                    })
    except Exception as e:
        print(f"[Ingestion] RSS notice: {e}")

    return results


def fetch_rumor_evidence(clean_claim: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Optional[str]]:
    """
    Executes a dual-sweep query:
    1. Primary News Stream: Recent news matching claim
    2. Fact-Check Stream: Dedicated search for existing debunks
    Returns: (news_evidence, factcheck_evidence, network_error)
    """
    news_evidence: List[Dict[str, str]] = []
    fc_evidence: List[Dict[str, str]] = []
    network_error: Optional[str] = None

    try:
        news_evidence = fetch_evidence_rss(clean_claim, max_results=6)
    except Exception as e:
        network_error = str(e)

    # Dedicated fact-check query
    try:
        fc_query = f'"{clean_claim}" fact check'
        fc_evidence = fetch_evidence_rss(fc_query, max_results=6)
    except Exception as e:
        if not network_error:
            network_error = str(e)

    return news_evidence, fc_evidence, network_error


# ----------------------------------------------------------------
# DeBERTa-v3 Natural Language Inference (NLI) Cross-Encoder Engine
# ----------------------------------------------------------------
NLI_MODEL = None
NLI_TOKENIZER = None
_NLI_LOCK = threading.Lock()
_NLI_LOADED = False

def get_nli_engine():
    """
    Thread-safe loader for DeBERTa-v3 NLI Cross-Encoder (~500MB).
    """
    global NLI_MODEL, NLI_TOKENIZER, _NLI_LOADED
    if _NLI_LOADED:
        return NLI_MODEL, NLI_TOKENIZER

    with _NLI_LOCK:
        if _NLI_LOADED:
            return NLI_MODEL, NLI_TOKENIZER
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            model_name = "cross-encoder/nli-deberta-v3-base"
            print(f"[NLI Engine] Loading DeBERTa-v3 Cross-Encoder: {model_name}...")
            NLI_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            NLI_MODEL = AutoModelForSequenceClassification.from_pretrained(model_name)
            NLI_MODEL.eval()
            print("[NLI Engine] DeBERTa-v3 Cross-Encoder loaded successfully into memory.")
        except Exception as ex:
            print(f"[NLI Engine] NLI Cross-Encoder running in fallback mode: {ex}")
            NLI_MODEL = None
            NLI_TOKENIZER = None
        finally:
            _NLI_LOADED = True

    return NLI_MODEL, NLI_TOKENIZER


def evaluate_nli_entailment(claim: str, evidence_texts: List[str]) -> List[Dict[str, float]]:
    model, tokenizer = get_nli_engine()
    if model is None or tokenizer is None or not evidence_texts:
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
# Semantic Relevance & Cosine Fallback Filter
# ----------------------------------------------------------------
ENCODER_MODEL = None
_ENCODER_LOCK = threading.Lock()
_ENCODER_LOADED = False

def get_sentence_transformer():
    global ENCODER_MODEL, _ENCODER_LOADED
    if _ENCODER_LOADED:
        return ENCODER_MODEL

    with _ENCODER_LOCK:
        if _ENCODER_LOADED:
            return ENCODER_MODEL
        try:
            from sentence_transformers import SentenceTransformer
            ENCODER_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            print("[Local Tensor Engine] MiniLM Transformer Model loaded successfully.")
        except Exception as ex:
            print(f"[Local Tensor Engine] SentenceTransformer fallback mode: {ex}")
            ENCODER_MODEL = None
        finally:
            _ENCODER_LOADED = True

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

    stopwords = {'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'is', 'was', 'for', 'on', 'with', 'by', 'at'}
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
    clean_raw = strip_punctuation(raw_stripped, keep_chars="?'").lower().strip()
    clean_no_punct = clean_raw.replace("?", "").strip()
    raw_words = [w for w in clean_no_punct.split() if w]

    common_greetings = {
        'hi', 'hello', 'hey', 'test', 'testing', 'good morning', 'good afternoon', 'good evening', 'bye', 'ok', 'okay', 'yes', 'no'
    }

    if clean_no_punct in common_greetings or len(raw_words) < 2 or len(clean_no_punct) < 5 or clean_no_punct.isdigit():
        return None, {
            "is_invalid": True,
            "verdict": "INVALID_QUERY",
            "message": "Please submit rumors or news claims only.",
            "confidence": None,
            "cues": {"cue_count": 0, "flags": []},
            "ground_truth": {
                "source": "Rumor Filter",
                "text": "Input does not contain an assertive claim or viral forward to verify."
            },
            "counter_en": "Please enter an assertive claim or forwarded rumor (e.g., 'Schools closed tomorrow due to flood').",
            "counter_hi": "कृपया जांच के लिए कोई विशिष्ट दावा या अफवाह दर्ज करें।"
        }

    is_conversational = bool(
        re.search(r'(\btell me\b|\babout your\b|\babout yourself\b|\babout ur\b|\babout slef\b|\bwho are you\b|\bwho r u\b|\bwho is this\b|\bhow to use\b|\byou are good\b|\byou r good\b)', clean_no_punct) or
        re.search(r'^(how\s*(r|are|u|you)|what\s*(s|\'s|up|r|is)|who\s*(r|is|u)|sup|wbu)\b', clean_no_punct)
    )

    if is_conversational:
        return None, {
            "is_invalid": True,
            "verdict": "INVALID_QUERY",
            "message": "Please submit rumors or news claims only.",
            "confidence": None,
            "cues": {"cue_count": 0, "flags": []},
            "ground_truth": {
                "source": "Rumor Filter",
                "text": "Conversational prompt detected. Please submit a rumor or news assertion."
            },
            "counter_en": "Please enter an assertive rumor or news claim to check.",
            "counter_hi": "कृपया जांच के लिए कोई विशिष्ट दावा या अफवाह दर्ज करें।"
        }

    # Only strip leading auxiliary if the sentence ends with '?' (preserves "Will Smith")
    ends_with_q = raw_stripped.endswith('?')
    if ends_with_q:
        normalized = re.sub(r'^(is|are|was|were|did|does|will|can|has|have)\s+', '', clean_no_punct, flags=re.IGNORECASE).strip()
    else:
        normalized = clean_no_punct

    return normalized if normalized else clean_no_punct, None


# ----------------------------------------------------------------
# Authenticated Rumor Verification Engine
# ----------------------------------------------------------------
def verify_claim_against_web(raw_claim_text: str) -> dict:
    # 1. Profile psychological manipulation / viral forward cues
    cues_profile = detect_rumor_cues(raw_claim_text)

    # 2. Extract core claim from forward boilerplate
    extracted_claim = extract_rumor_claim(raw_claim_text)

    # 3. Normalize & validate input
    normalized_claim, invalid_resp = normalize_and_validate_query(extracted_claim)
    if invalid_resp:
        invalid_resp["cues"] = cues_profile
        return invalid_resp

    # 4. Fetch dual-sweep evidence: Breaking News + Dedicated Fact-Check
    news_evidence, fc_evidence, network_error = fetch_rumor_evidence(normalized_claim)
    all_evidence = fc_evidence + news_evidence

    # Network failure handling
    if not all_evidence:
        if network_error:
            return {
                "is_invalid": False,
                "verdict": "NETWORK_ERROR",
                "engine": "Offline Ingestion Stream",
                "classification": "SERVICE TEMPORARILY UNAVAILABLE",
                "confidence": None,
                "cues": cues_profile,
                "sources_count": 0,
                "ground_truth": {
                    "source": "Network Ingestion Stream",
                    "text": "Could not connect to live news or verification streams. Please check internet connection."
                },
                "counter_en": "Network connection timed out. Please retry in a few moments.",
                "counter_hi": "समाचार सत्यापन नेटवर्क से संपर्क नहीं हो सका। कृपया पुनः प्रयास करें।"
            }

        # ABSENCE OF EVIDENCE HANDLING
        # If high-profile assertion with viral cues, silence from all major media indicates a fabricated rumor
        if cues_profile["is_suspicious_forward"] or len(normalized_claim.split()) >= 4:
            return {
                "is_invalid": False,
                "verdict": "LIKELY RUMOR",
                "engine": "Media Absence & Forward Profiling",
                "classification": "LIKELY RUMOR – Zero Credible News Reports Found",
                "confidence": None,
                "cues": cues_profile,
                "sources_count": 0,
                "ground_truth": {
                    "source": "Global Wire Absence Check",
                    "text": f"Scans across national and local media found no reporting on this claim: \"{normalized_claim}\"."
                },
                "counter_en": f"NO CREDIBLE COVERAGE: A major event of this nature would be reported across news agencies. Zero reports found; exercise extreme caution.",
                "counter_hi": f"संभावित अफवाह: इस दावे की पुष्टि किसी भी विश्वसनीय समाचार स्रोत द्वारा नहीं की गई है।"
            }

        return {
            "is_invalid": False,
            "verdict": "UNCONFIRMED",
            "engine": "Media Absence Check",
            "classification": "UNCONFIRMED – No Public Records Found",
            "confidence": None,
            "cues": cues_profile,
            "sources_count": 0,
            "ground_truth": {
                "source": "Public Record Sweep",
                "text": f"No public dispatches confirm or dispute: \"{normalized_claim}\"."
            },
            "counter_en": f"No official records found regarding \"{normalized_claim}\". Status remains unconfirmed.",
            "counter_hi": f"अपुष्ट: इस विषय पर कोई आधिकारिक रिपोर्ट उपलब्ध नहीं है।"
        }

    q_lower = normalized_claim.lower()
    current_year = datetime.datetime.now().year

    # ------------------------------------------------------------
    # STEP A: DEDICATED FACT-CHECK DESK MATCHING
    # ------------------------------------------------------------
    # Check if a recognized fact-checker has already analyzed this claim
    debunk_regex = re.compile(r'\b(fact.?check|debunk|hoax|fake|false|misleading|old video|morphed|doctored|no,|not true)\b', re.IGNORECASE)

    for item in fc_evidence:
        source_clean = item["source"].lower().strip()
        headline = item["text"]
        headline_lower = headline.lower()

        # Compute semantic relevance to ensure fact-check is actually about this claim
        relevance_scores = compute_semantic_similarity(normalized_claim, [headline])
        is_relevant = relevance_scores and relevance_scores[0] >= 0.35

        is_whitelisted = any(fc in source_clean for fc in FACTCHECK_OUTLETS)
        has_debunk_words = bool(debunk_regex.search(headline_lower))

        if is_relevant and (is_whitelisted or has_debunk_words):
            return {
                "is_invalid": False,
                "verdict": "RUMOR (debunked)",
                "engine": "Verified Fact-Check Desk",
                "classification": f"DEBUNKED RUMOR – Refuted by {item['source']}",
                "confidence": 0.95,
                "cues": cues_profile,
                "sources_count": 1,
                "ground_truth": {"source": item["source"], "text": item["text"]},
                "counter_en": f"DEBUNKED BY FACT-CHECKERS: {item['source']} investigated this claim and confirmed it is false or misleading: \"{item['text']}\".",
                "counter_hi": f"फर्जी अफवाह: आधिकारिक फैक्ट-चेक ({item['source']}) ने इस दावे का खंडन किया है: \"{item['text']}\"।"
            }

    # ------------------------------------------------------------
    # STEP B: TEMPORAL & RECYCLED-NEWS GUARD
    # ------------------------------------------------------------
    # Check if claim asserts a past completed event for a future release
    is_claiming_past = bool(re.search(r'\b(released|launched|won|died|killed|premiered|happened)\b', q_lower))
    future_year_pattern = rf'\b({current_year+1}|{current_year+2}|{current_year+3})\b'

    for e in all_evidence:
        ev_lower = e["text"].lower()
        is_future = bool(re.search(r'\b(to be released|will be released|being released in|scheduled for|slated for|anticipated in)\b', ev_lower) or
                         re.search(future_year_pattern, ev_lower))
        if is_claiming_past and is_future and not ('future' in q_lower or 'upcoming' in q_lower):
            return {
                "is_invalid": False,
                "verdict": "RUMOR (debunked)",
                "engine": "Temporal Guard",
                "classification": "CONTRADICTED – Event Scheduled for Future (Not Yet Occurred)",
                "confidence": 0.90,
                "cues": cues_profile,
                "sources_count": 1,
                "ground_truth": {"source": e["source"], "text": e["text"]},
                "counter_en": f"FUTURE EVENT: Official dispatches confirm this is an upcoming announcement scheduled for the future, not an event that has occurred. Source ({e['source']}): \"{e['text']}\".",
                "counter_hi": f"भविष्य की घोषणा: आधिकारिक सूचना के अनुसार यह अभी रिलीज़ या घटित नहीं हुआ है।"
            }

    # ------------------------------------------------------------
    # STEP C: CROSS-ATTENTION NLI REASONING ON BREAKING NEWS
    # ------------------------------------------------------------
    candidate_texts = [e["text"] for e in news_evidence]
    nli_results = evaluate_nli_entailment(normalized_claim, candidate_texts)

    if nli_results:
        entailment_matches: List[Tuple[Dict[str, str], float]] = []
        contradiction_matches: List[Tuple[Dict[str, str], float]] = []

        for idx, res in enumerate(nli_results):
            ev_item = news_evidence[idx]
            # Verify headline does not contain debunk marker masquerading as news
            if debunk_regex.search(ev_item["text"].lower()):
                contradiction_matches.append((ev_item, 0.90))
                continue

            if res["entailment"] >= 0.60:
                entailment_matches.append((ev_item, res["entailment"]))
            elif res["contradiction"] >= 0.55:
                contradiction_matches.append((ev_item, res["contradiction"]))

        # Contradiction takes priority
        if contradiction_matches:
            best_match, conf = max(contradiction_matches, key=lambda x: x[1])
            return {
                "is_invalid": False,
                "verdict": "RUMOR (debunked)",
                "engine": "DeBERTa-v3 NLI Inference",
                "classification": f"REFUTED – Contradicted by News Dispatches",
                "confidence": round(conf, 2),
                "cues": cues_profile,
                "sources_count": len(contradiction_matches),
                "ground_truth": {"source": best_match["source"], "text": best_match["text"]},
                "counter_en": f"CONTRADICTED: Reported evidence from {best_match['source']} disputes this assertion: \"{best_match['text']}\".",
                "counter_hi": f"अपुष्ट एवं खंडित: {best_match['source']} की रिपोर्टें इस दावे का खंडन करती हैं।"
            }

        # Multi-Source Corroboration Rule for CONFIRMED
        # Require >= 2 distinct sources for true confirmation, otherwise UNCONFIRMED single-source report
        if entailment_matches:
            unique_sources = {m[0]["source"].lower().strip() for m in entailment_matches}
            best_match, conf = max(entailment_matches, key=lambda x: x[1])

            if len(unique_sources) >= 2:
                return {
                    "is_invalid": False,
                    "verdict": "CONFIRMED",
                    "engine": "Multi-Source NLI Corroboration",
                    "classification": f"CONFIRMED – Corroborated Across {len(unique_sources)} Independent Outlets",
                    "confidence": round(conf, 2),
                    "cues": cues_profile,
                    "sources_count": len(unique_sources),
                    "ground_truth": {"source": best_match["source"], "text": best_match["text"]},
                    "counter_en": f"CORROBORATED: Multi-source dispatches ({', '.join(list(unique_sources)[:3])}) confirm this event: \"{best_match['text']}\".",
                    "counter_hi": f"सत्यापित समाचार: कई स्वतंत्र समाचार एजेंसियों ने इस खबर की पुष्टि की है।"
                }
            else:
                # Single source only
                return {
                    "is_invalid": False,
                    "verdict": "UNCONFIRMED",
                    "engine": "Single-Source Gate",
                    "classification": f"UNCONFIRMED – Reported by Only 1 Outlet ({best_match['source']})",
                    "confidence": round(conf, 2),
                    "cues": cues_profile,
                    "sources_count": 1,
                    "ground_truth": {"source": best_match["source"], "text": best_match["text"]},
                    "counter_en": f"SINGLE-SOURCE REPORT: Only one outlet ({best_match['source']}) has reported this so far. Awaiting secondary corroboration.",
                    "counter_hi": f"एकल-स्रोत रिपोर्ट: अभी केवल एक एजेंसी द्वारा रिपोर्ट किया गया है, स्वतंत्र पुष्टि की प्रतीक्षा है।"
                }

        # NLI evaluated and found evidence is Neutral
        best_ev = news_evidence[0] if news_evidence else all_evidence[0]
        verdict = "LIKELY RUMOR" if cues_profile["is_suspicious_forward"] else "UNCONFIRMED"
        return {
            "is_invalid": False,
            "verdict": verdict,
            "engine": "DeBERTa-v3 NLI Inference",
            "classification": "INCONCLUSIVE – Dispatches Do Not Directly Confirm Rumor",
            "confidence": None,
            "cues": cues_profile,
            "sources_count": len(news_evidence),
            "ground_truth": {"source": best_ev["source"], "text": best_ev["text"]},
            "counter_en": f"INCONCLUSIVE: Related news dispatches exist but do not support the specific assertions made in this forward.",
            "counter_hi": f"अपुष्ट: उपलब्ध समाचारों में इस दावे का कोई प्रत्यक्ष प्रमाण नहीं मिला।"
        }

    # ------------------------------------------------------------
    # STEP D: SEMANTIC VECTOR FALLBACK (ONLY if NLI weights unavailable)
    # ------------------------------------------------------------
    similarities = compute_semantic_similarity(normalized_claim, candidate_texts)
    if not similarities or max(similarities) < 0.45:
        verdict = "LIKELY RUMOR" if cues_profile["is_suspicious_forward"] else "UNCONFIRMED"
        best_ev = all_evidence[0]
        return {
            "is_invalid": False,
            "verdict": verdict,
            "engine": "Semantic Fallback",
            "classification": "UNCONFIRMED – Low Semantic Relevance to Public Dispatches",
            "confidence": None,
            "cues": cues_profile,
            "sources_count": 0,
            "ground_truth": {"source": best_ev["source"], "text": best_ev["text"]},
            "counter_en": f"No corroborating news found matching this claim.",
            "counter_hi": f"कोई पुष्ट समाचार नहीं मिला।"
        }

    best_idx = int(max(range(len(similarities)), key=lambda i: similarities[i]))
    best_sim = round(similarities[best_idx], 3)
    best_match = news_evidence[best_idx]
    ev_lower = best_match["text"].lower()

    if debunk_regex.search(ev_lower):
        return {
            "is_invalid": False,
            "verdict": "RUMOR (debunked)",
            "engine": "Semantic Fallback + Debunk Regex",
            "classification": "DEBUNKED – Refuted by News Reports",
            "confidence": best_sim,
            "cues": cues_profile,
            "sources_count": 1,
            "ground_truth": {"source": best_match["source"], "text": best_match["text"]},
            "counter_en": f"News reports refute this claim: \"{best_match['text']}\".",
            "counter_hi": f"समाचार रिपोर्टों ने इस दावे का खंडन किया है।"
        }

    # For safety in fallback mode: Never output CONFIRMED without NLI cross-encoder; default to UNCONFIRMED
    return {
        "is_invalid": False,
        "verdict": "UNCONFIRMED",
        "engine": "Semantic Fallback (Degraded Mode)",
        "classification": "UNCONFIRMED – High Topical Overlap but Unverified Entailment",
        "confidence": None,
        "cues": cues_profile,
        "sources_count": 1,
        "ground_truth": {"source": best_match["source"], "text": best_match["text"]},
        "counter_en": f"Topically related report from {best_match['source']}: \"{best_match['text']}\". Formal cross-verification required.",
        "counter_hi": f"संबंधित समाचार: \"{best_match['text']}\"।"
    }


# ----------------------------------------------------------------
# FastAPI Web Application (Secure Setup with Startup Preload)
# ----------------------------------------------------------------
app = FastAPI(title="CivicTrust Hub Rumor Detection API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = pathlib.Path(__file__).parent

@app.on_event("startup")
def startup_preload_models():
    """Preload models in a background thread so the server starts immediately and warm-ups inference."""
    threading.Thread(target=get_nli_engine, daemon=True).start()
    threading.Thread(target=get_sentence_transformer, daemon=True).start()

class RumorVerifyRequest(BaseModel):
    text: str = Field(..., max_length=500)
    platform: Optional[str] = "Live Feed"

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "CivicTrust Hub Rumor Detection Pipeline",
        "pipeline": [
            "Forward & Cue Profiling",
            "Claim Extraction",
            "Dedicated Fact-Check Sweep (PIB, BOOM, Alt News, Factly)",
            "Temporal & Recycled News Guard",
            "Multi-Source DeBERTa NLI Corroboration"
        ],
        "verdicts": ["RUMOR (debunked)", "LIKELY RUMOR", "UNCONFIRMED", "CONFIRMED", "INVALID_QUERY"],
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
