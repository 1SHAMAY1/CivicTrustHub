# CivicTrustHub 🛡️

> **Fast, Real-Time Civic Rumor & Disinformation Corroboration Engine**  
> Zero dummy data. Zero costly external LLM APIs. Real-time news stream ingestion with local PyTorch MiniLM tensor embeddings.

[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![GitHub Repository](https://img.shields.io/badge/GitHub-1SHAMAY1%2FCivicTrustHub-blue?logo=github)](https://github.com/1SHAMAY1/CivicTrustHub)
[![Live Web Application](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-green)](https://1shamay1.github.io/CivicTrustHub/)

---

## 📌 Problem Statement
Unverified rumors, fabricated news, and coordinated disinformation proliferate across social media and messaging platforms (WhatsApp, X, Telegram) within minutes—spanning public policy, celebrity gossip, civic advisories, elections, and crisis events. By the time official statements or traditional fact-checkers release a debunker, false narratives have already gone viral, skewing public perception, eroding social trust, and inciting unnecessary panic.

---

## 💡 Solution
**CivicTrustHub** is an ultra-fast, zero-LLM-cost verification engine that delivers instant truth verdicts on viral online claims and everyday social media rumors. It couples real-time web and news stream ingestion (Google News RSS & Wikipedia) with local PyTorch MiniLM tensor embeddings and a multi-stage linguistic evaluation pipeline (polarity alignment, temporal guards, and conversational prompt filtering). Citizens and authorities receive immediate, verifiable source evidence without proprietary API fees or stale databases.

---

## 🛡️ Multi-Layer Precision & Robustness Engine
To prevent hallucinated verdicts and bag-of-words pitfalls, CivicTrustHub processes claims through 4 robust analytical layers:
1. **Conversational & Non-Assertion Guard**: Detects questions (`"how to..."`, `"what is..."`), 2nd-person chat prompts (`"tell me about yourself"`), and conversational praise, guiding the user to input assertive claims only.
2. **Directional Polarity & Negation Logic**: Distinguishes affirmative claims from negative assertions (`"not married"`, `"never happened"`), evaluating sign alignment between the claim and the retrieved evidence.
3. **Temporal Future-Tense Guard**: Analyzes publication dates and future modal qualifiers (`"scheduled for 2026"`, `"upcoming release"`), preventing unreleased upcoming media or events from being falsely marked as completed.
4. **Calibrated Semantic Tensor Scoring**: Combines subject-action embeddings with cosine similarity thresholds to dynamically scale confidence scores without arbitrary hardcoded jumps.

---

## 👥 Target Audience & Why They Use It
- **Everyday Citizens & Social Media Users**: 
  - *Why they use it*: To get a 2-second sanity check before forwarding scary or suspicious WhatsApp/Telegram messages to family groups, stopping panic at the source.
- **National Security, Law Enforcement & Disaster Management (NDRF/SDMA)**: 
  - *Why they use it*: Early warning detection of viral synthetic rumors and social unrest narratives, preventing civilian stampedes and safeguarding critical infrastructure.
- **High-Profile Public Figures & Government PR Desks**: 
  - *Why they use it*: To instantly spot deepfake claims, fabricated press releases, or impersonation rumors targeted at them before they damage public trust or influence markets.
- **Fact-Checkers & Field Journalists**: 
  - *Why they use it*: Automated first-pass triage to separate obvious hoaxes from verifiable breaking incidents without manual search latency.

---

## 🏗️ Architecture & Data Flow

```mermaid
flowchart TD
    User([Citizens / Target Audience]) -->|Enter claim or viral rumor| UI[Tactile Skeuomorphic UI - index.html]
    
    subgraph PreValidation ["Input Validation & Edge Filter"]
        UI --> EdgeFilter{Is Rumor or Edge Case?}
        EdgeFilter -->|Greeting / Number / Too Short| Warning["Alert: 'Please check rumors only.'"]
    end

    subgraph RuntimeDispatch ["Execution Strategy Switch"]
        EdgeFilter -->|Valid Assertive Claim| EnvSwitch{Host Environment?}
        EnvSwitch -->|Localhost / Native App| BackendAPI["FastAPI: POST /api/rumors/verify"]
        EnvSwitch -->|GitHub Pages / Web App| ClientLiveEngine[Browser Real-Time Web Engine]
    end

    subgraph ServerBackend ["Server Engine (main.py)"]
        BackendAPI --> ServerEdge{Backend Edge Validator}
        ServerEdge -->|Invalid Query| FastReject["Return 200: 'Please check rumors only.'"]
        ServerEdge -->|Valid Claim| LiveSweep[Live Ingestion: Google News RSS + Wiki API]
        LiveSweep --> TensorMatcher["Semantic Tensor Matcher (PyTorch MiniLM / TF-IDF Vector)"]
        TensorMatcher --> PolarityTemporal["Polarity Alignment & Temporal Guard Engine"]
        PolarityTemporal --> ScoreEngine["Cosine Similarity & Panic Index Classifier"]
    end

    subgraph BrowserEngine ["In-Browser Engine"]
        ClientLiveEngine --> WikiSearch[Wikimedia Real-Time REST Search]
        WikiSearch --> PolarityTemporalClient["Polarity & Future-Tense Analysis"]
        PolarityTemporalClient --> KeywordDensity[Calibrated Confidence Scorer]
    end

    ScoreEngine --> VerdictCard[Verdict Card: TRUE / FALSE + Proof Evidence]
    KeywordDensity --> VerdictCard
    VerdictCard --> UI
```

---

## ⚙️ Technology Stack

| Layer | Technologies Used | Description & Licensing |
| :--- | :--- | :--- |
| **High-Precision NLI Engine** | **DeBERTa-v3 NLI Cross-Encoder** (`cross-encoder/nli-deberta-v3-base`) | 500MB cross-attention inference model delivering direct Entailment vs. Contradiction classification. Apache 2.0. |
| **Tensor Embedding Engine** | **PyTorch**, **Sentence-Transformers** (`all-MiniLM-L6-v2`) | Dense 384-dimensional semantic tensor embedding model. Apache 2.0. |
| **Vector Fallback Engine** | **NumPy**, Vectorized TF-IDF Cosine Calculator | Ultra-lightweight tensor cosine similarity engine requiring zero external weights. |
| **Backend Framework** | **Python 3.10+**, **FastAPI**, **Uvicorn**, **Pydantic** | Asynchronous REST API serving verification endpoints. MIT License. |
| **Live Web Ingestion** | **Google News RSS**, **Wikimedia Action API** | 100% live streaming data ingestion. Zero hardcoded mock databases. |
| **Frontend & UI** | **Vanilla JavaScript (ES6+)**, **Tailwind CSS** | Skeuomorphic tactile interface with real-time responsive cards. MIT License. |

---

## 👨‍💻 Team & Contributions

This project was built collaboratively by a 3-person team:

- **Shankharaj Datta** – *System Architect*  
  System design, dual-engine execution strategy (FastAPI + client-side web engine), tensor embedding pipeline design, Private Network Access security compliance, and licensing.
- **Soumendu Das** – *Backend Engineer*  
  Real-time Google News RSS & Wikipedia live ingestion streams, PyTorch MiniLM / vector cosine tensor calculation, panic index metric, and backend edge-case verification logic.
- **Ankan Laskar** – *Frontend Engineer*  
  Tactile skeuomorphic UI design with TailwindCSS, live search event pipeline, WhatsApp-style search interactions, dynamic evidence card rendering, and client-side validation.

---

## 📜 Compliance & 100% Open-Source Licensing

This project is completely open source under the **[MIT License](LICENSE)**.

Every single dependency, model, and framework utilized in this project is **100% free, open-source software (FOSS)** with permissive licensing (MIT, Apache 2.0, or BSD). No proprietary software, paid API keys, or closed-source platforms are required.

### Third-Party Open-Source Attributions:
- **Sentence-Transformers (`all-MiniLM-L6-v2`)**: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) – Open-source embedding model by HuggingFace / UKPLab.
- **PyTorch**: [PyTorch License (Modified BSD-style)](https://github.com/pytorch/pytorch/blob/main/LICENSE) – Open-source ML framework under Linux Foundation.
- **FastAPI**: [MIT License](https://github.com/tiangolo/fastapi/blob/master/LICENSE) – Open-source web framework by Sebastián Ramírez.
- **Uvicorn**: [BSD 3-Clause License](https://github.com/encode/uvicorn/blob/master/LICENSE.md) – Open-source ASGI web server by Encode OSS.
- **Pydantic**: [MIT License](https://github.com/pydantic/pydantic/blob/main/LICENSE) – Open-source data validation library by Pydantic Services Inc.
- **Tailwind CSS**: [MIT License](https://github.com/tailwindlabs/tailwindcss/blob/master/LICENSE) – Open-source utility CSS framework by Tailwind Labs Inc.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Pip package manager

### 1. Clone the Repository
```bash
git clone https://github.com/1SHAMAY1/CivicTrustHub.git
cd CivicTrustHub
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Optional for GPU / Dense MiniLM acceleration)*:
```bash
pip install sentence-transformers torch
```

### 3. Run the Backend Server
```bash
uvicorn main:app --reload --port 8000
```
Open your browser at `http://127.0.0.1:8000` to interact with the live engine.

### 4. Or Run Static Frontend Directly
You can also open `index.html` directly or visit the [Live GitHub Pages Demo](https://1shamay1.github.io/CivicTrustHub/).
