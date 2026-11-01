# ==============================================================
# CivicTrust Hub – FastAPI Backend
# Serves:
#   GET  /api/health             → ping / status
#   POST /api/rumors/verify      → semantic rumor verification
#   GET  /                       → serves frontend/index.html
# ==============================================================

from __future__ import annotations
import json
import math
import pathlib
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ----------------------------------------------------------------
# Official bulletins (ground truth used for rumor verification)
# ----------------------------------------------------------------
OFFICIALS: list[dict] = [
    {
        "source": "State Disaster Management Authority (SDMA)",
        "text": "All reservoir gates operating under regulated discharge. No unannounced opening. Water levels within safety margin.",
    },
    {
        "source": "Municipal Corporation Health Department",
        "text": "Continuous water testing conducted. Residual chlorine at healthy 0.5 ppm. Zero cholera or epidemic cases in hospitals.",
    },
    {
        "source": "Metro Rail Corporation",
        "text": "Metro operations running on scheduled frequency. Track pumps functioning normally. No station closures.",
    },
    {
        "source": "Medical Superintendent Bureau",
        "text": "Hospital holds 72 hours of uninterrupted liquid oxygen reserves in cryogenic tanks. No shortages exist.",
    },
    {
        "source": "Municipal Commissioner IT Wing",
        "text": "Traffic cameras calibrated solely for license plate recognition and traffic flow. No facial recognition or banking integration.",
    },
    {
        "source": "District Collector Office",
        "text": "Relief camps are fully equipped with power, meals and medicines. Situation is under control.",
    },
]

# ----------------------------------------------------------------
# Attempt to load the MiniLM encoder; fall back to keyword heuristic
# ----------------------------------------------------------------
ENCODER = None

def _try_load_encoder():
    global ENCODER
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model_dir = pathlib.Path(__file__).parent / "miniLM_quantized"
        if model_dir.is_dir():
            m = SentenceTransformer(str(model_dir))
        else:
            m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        official_texts = [b["text"] for b in OFFICIALS]
        embs = m.encode(official_texts, normalize_embeddings=True)

        def _encode(texts):
            return m.encode(texts, normalize_embeddings=True)

        ENCODER = {"model": m, "official_embs": embs, "encode": _encode}
        print("[CivicTrust] ✅ MiniLM encoder loaded.")
    except Exception as ex:
        print(f"[CivicTrust] ⚠️  Encoder unavailable ({ex}). Using keyword fallback.")
        ENCODER = None


_try_load_encoder()


# ----------------------------------------------------------------
# Core verification logic
# ----------------------------------------------------------------
def _keyword_similarity(text: str, bulletin: str) -> float:
    """Simple word-overlap heuristic – used when MiniLM is unavailable."""
    words_a = set(text.lower().replace(",", " ").replace(".", " ").split())
    words_b = set(bulletin.lower().replace(",", " ").replace(".", " ").split())
    # filter very short words
    words_a = {w for w in words_a if len(w) > 3}
    words_b = {w for w in words_b if len(w) > 3}
    if not words_a or not words_b:
        return 0.15
    intersection = words_a & words_b
    union = words_a | words_b
    jaccard = len(intersection) / len(union)
    # scale into a cosine-like range [0.12, 0.90]
    return min(0.90, max(0.12, jaccard * 2.2))


def verify_rumor(text: str) -> dict:
    """Return the best-matching official bulletin and derived metrics."""
    best_sim = 0.0
    best_bulletin = OFFICIALS[0]

    if ENCODER is not None:
        import numpy as np
        rumor_vec = ENCODER["encode"]([text])[0]
        sims = ENCODER["official_embs"] @ rumor_vec
        best_idx = int(sims.argmax())
        best_sim = float(sims[best_idx])
        best_bulletin = OFFICIALS[best_idx]
    else:
        for bulletin in OFFICIALS:
            sim = _keyword_similarity(text, bulletin["text"])
            if sim > best_sim:
                best_sim = sim
                best_bulletin = bulletin

    is_unverified = best_sim < 0.40
    panic_index = round(min(99, max(10, (1 - best_sim) * 100)))

    if is_unverified:
        classification = "UNVERIFIED RUMOR – No official confirmation found."
        counter_en = (
            f'FACT CHECK: The claim "{text}" contradicts official data. '
            f'According to {best_bulletin["source"]}: '
            f'"{best_bulletin["text"]}" Please verify before sharing.'
        )
        counter_hi = (
            "खंडन: यह दावा आधिकारिक बुलेटिन से मेल नहीं खाता है। "
            "स्थिति सामान्य और नियंत्रित है। कृपया अफवाहें न फैलाएं।"
        )
    else:
        classification = "VERIFIED – Matches official bulletin."
        counter_en = (
            f'VERIFIED: This aligns with official release from {best_bulletin["source"]}: '
            f'"{best_bulletin["text"]}"'
        )
        counter_hi = (
            "सत्यापित: यह सूचना आधिकारिक सरकारी पुष्टि के अनुरूप है। "
            "यह विश्वसनीय स्रोत से मिली जानकारी है।"
        )

    return {
        "classification": classification,
        "cosine_similarity": round(best_sim, 3),
        "panic_index": panic_index,
        "is_unverified": is_unverified,
        "ground_truth": best_bulletin,
        "counter_en": counter_en,
        "counter_hi": counter_hi,
    }


# ----------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------
app = FastAPI(title="CivicTrust Hub API", version="1.0.0")

# Allow browser requests from the same machine (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the front-end static files
FRONTEND_DIR = pathlib.Path(__file__).parent / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ----------------------------------------------------------------
# Pydantic request model
# ----------------------------------------------------------------
class RumorVerifyRequest(BaseModel):
    text: str
    platform: Optional[str] = "Unknown"


# ----------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------
@app.get("/api/health")
def health():
    """Ping endpoint – the front-end checks this to decide engine mode."""
    return {
        "status": "ok",
        "engine": "MiniLM Encoder" if ENCODER else "Keyword Fallback",
        "version": "1.0.0",
    }


@app.post("/api/rumors/verify")
def rumors_verify(body: RumorVerifyRequest):
    """
    POST { text: str, platform: str }
    Returns semantic verification result consumed by the front-end's
    testCustomRumor() function.
    """
    result = verify_rumor(body.text)
    return JSONResponse(content=result)


@app.get("/api/officials")
def get_officials():
    """Return the list of official ground-truth bulletins."""
    return {"officials": OFFICIALS}


@app.get("/")
def serve_index():
    """Serve the front-end index.html when visiting http://localhost:8000"""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return JSONResponse(
        {"message": "CivicTrust Hub API is running. Frontend not found at /frontend/index.html"},
        status_code=404,
    )

