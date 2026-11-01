"""
Unit and integration tests for CivicTrustHub core engine.
Directly imports main.py to verify real normalization, edge-case rejection,
Unicode support, network resilience, and 3-state verdict classification.
"""
from unittest.mock import patch
import main

# 1. Linguistic & Name Preservation Tests
def test_will_smith_name_not_stripped():
    q, err = main.normalize_and_validate_query("Will Smith released new movie")
    assert err is None
    assert q.startswith("will smith")

def test_interrogative_stripped_when_ending_with_question_mark():
    q, err = main.normalize_and_validate_query("Is Delhi airport flooded?")
    assert err is None
    assert q == "delhi airport flooded"

def test_contraction_negation_preserved():
    q, err = main.normalize_and_validate_query("Delhi airport isn't flooded")
    assert err is None
    assert "isn't" in q

def test_marriott_not_matched_as_marriage():
    q, err = main.normalize_and_validate_query("Marriott hotel opened in Delhi")
    assert err is None
    assert not main.re.search(r'\bmarri(ed|age)\b|\bwedding\b', q)

# 2. Strict Unicode & Indic Script Preservation
def test_hindi_script_preserved():
    q, err = main.normalize_and_validate_query("दिल्ली में भारी बारिश और बाढ़ का अलर्ट")
    assert err is None
    assert "दिल्ली" in q
    assert "बाढ़" in q

def test_bengali_script_preserved():
    q, err = main.normalize_and_validate_query("পশ্চিমবঙ্গে কাল স্কুল ছুটি ঘোষণা")
    assert err is None
    assert "পশ্চিমবঙ্গে" in q
    assert "স্কুল" in q

# 3. Conversational Chitchat & Typo Filtering
def test_greetings_rejected():
    for greeting in ["hi", "hello", "hey", "good morning", "ok"]:
        _, err = main.normalize_and_validate_query(greeting)
        assert err is not None
        assert err["verdict"] == "INVALID_QUERY"

def test_conversational_prompts_and_typos_rejected():
    prompts = [
        "well tell me about your slef",
        "how u doing",
        "who are you",
        "how to use this",
        "you are good",
        "what is your name"
    ]
    for prompt in prompts:
        _, err = main.normalize_and_validate_query(prompt)
        assert err is not None
        assert err["verdict"] == "INVALID_QUERY"

def test_pure_numbers_rejected():
    _, err = main.normalize_and_validate_query("123456789")
    assert err is not None
    assert err["verdict"] == "INVALID_QUERY"

# 4. Temporal Current Year vs Future Year Tests
def test_current_year_not_flagged_as_future():
    # Present-day 2026 events must NOT be marked as future announcements
    mock_evidence = [{
        "source": "Seismological Dispatches",
        "text": "Earthquake magnitude 6.2 reported on 15 March 2026 in coastal region."
    }]
    with patch("main.fetch_live_web_evidence", return_value=(mock_evidence, None)):
        res = main.verify_claim_against_web("Earthquake reported in 2026")
        assert res["verdict"] != "CONTRADICTED" or "Future" not in res["classification"]

def test_future_release_flagged_as_contradicted():
    mock_evidence = [{
        "source": "Film Industry Wire",
        "text": "The sequel is scheduled for release in December 2028."
    }]
    with patch("main.fetch_live_web_evidence", return_value=(mock_evidence, None)):
        res = main.verify_claim_against_web("Sequel released")
        assert res["verdict"] == "CONTRADICTED"
        assert "Future" in res["classification"]

# 5. Network Error & Resilience Tests
def test_network_timeout_reported_as_unavailable():
    with patch("main.fetch_live_web_evidence", return_value=([], "Connection timed out after 6000ms")):
        res = main.verify_claim_against_web("Cyclone warning issued for Odisha")
        assert res["verdict"] == "NETWORK_ERROR"
        assert res["classification"] == "SERVICE TEMPORARILY UNAVAILABLE"
        assert res["confidence"] is None

def test_no_evidence_found_reported_as_unverified():
    with patch("main.fetch_live_web_evidence", return_value=([], None)):
        res = main.verify_claim_against_web("Completely fictional rumor with zero coverage 999xyz")
        assert res["verdict"] == "UNVERIFIED"
        assert res["confidence"] is None

# 6. NLI Isolation Test (Neutral stops at UNVERIFIED and does not fall back to similarity)
def test_nli_neutral_returns_unverified_without_fallback():
    mock_evidence = [{
        "source": "City Transport Desk",
        "text": "Delhi airport inaugurated a brand new domestic terminal today."
    }]
    # Simulate NLI evaluating this as Neutral (unrelated)
    mock_nli = [{"entailment": 0.10, "contradiction": 0.05, "neutral": 0.85}]
    with patch("main.fetch_live_web_evidence", return_value=(mock_evidence, None)):
        with patch("main.evaluate_nli_entailment", return_value=mock_nli):
            res = main.verify_claim_against_web("Delhi airport flooded")
            # Must be UNVERIFIED, NOT marked VERIFIED by word overlap
            assert res["verdict"] == "UNVERIFIED"
            assert res["confidence"] is None

# 7. Debunk Headline Detection Test
def test_debunk_headline_flagged_as_contradicted():
    mock_evidence = [{
        "source": "PIB Fact Check",
        "text": "Fact check: Viral claim that schools are closed tomorrow is fake and misleading."
    }]
    mock_nli = [{"entailment": 0.70, "contradiction": 0.20, "neutral": 0.10}]
    with patch("main.fetch_live_web_evidence", return_value=(mock_evidence, None)):
        with patch("main.evaluate_nli_entailment", return_value=mock_nli):
            res = main.verify_claim_against_web("Schools closed tomorrow")
            assert res["verdict"] == "CONTRADICTED"
            assert "Debunk" in res["classification"] or "Fact" in res["classification"]
