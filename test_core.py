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

def test_marriott_not_matched_as_marriage():
    q, err = main.normalize_and_validate_query("Marriott hotel opened in Delhi")
    assert err is None
    assert not main.re.search(r'\bmarri(ed|age)\b|\bwedding\b', q)

# 2. Unicode & Non-ASCII Regional Language Tests
def test_hindi_script_accepted():
    q, err = main.normalize_and_validate_query("दिल्ली में भारी बारिश और बाढ़ का अलर्ट")
    assert err is None
    assert len(q) > 10

def test_bengali_script_accepted():
    q, err = main.normalize_and_validate_query("পশ্চিমবঙ্গে কাল স্কুল ছুটি ঘোষণা")
    assert err is None
    assert len(q) > 10

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
    # An earthquake reported in 2026 is a present-day event, not a future release
    mock_evidence = [{
        "source": "Seismological Dispatches",
        "text": "Earthquake magnitude 6.2 reported on 15 March 2026 in coastal region."
    }]
    with patch("main.fetch_live_web_evidence", return_value=(mock_evidence, None)):
        res = main.verify_claim_against_web("Earthquake reported in 2026")
        assert res["verdict"] != "CONTRADICTED" or "Future" not in res["classification"]

def test_future_release_flagged():
    # A movie slated for 2028 is unreleased
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
    # A network failure must NOT be fabricated as "zero credible news / false rumor"
    with patch("main.fetch_live_web_evidence", return_value=([], "Connection timed out after 5000ms")):
        res = main.verify_claim_against_web("Cyclone warning issued for Odisha")
        assert res["verdict"] == "NETWORK_ERROR"
        assert res["classification"] == "SERVICE TEMPORARILY UNAVAILABLE"
        assert res["confidence"] is None

def test_no_evidence_found_reported_as_unverified():
    # When queries yield zero records, it must be UNVERIFIED, not FALSE
    with patch("main.fetch_live_web_evidence", return_value=([], None)):
        res = main.verify_claim_against_web("Completely fictional rumor with zero coverage 999xyz")
        assert res["verdict"] == "UNVERIFIED"
        assert res["confidence"] is None

# 6. 3-State Verdict Integrity
def test_verified_verdict_structure():
    mock_evidence = [{
        "source": "Official News Bureau",
        "text": "Official dispatches confirm classes will remain closed tomorrow due to heavy rain."
    }]
    with patch("main.fetch_live_web_evidence", return_value=(mock_evidence, None)):
        res = main.verify_claim_against_web("Schools closed tomorrow due to rain")
        assert res["verdict"] in ("VERIFIED", "UNVERIFIED")
        assert "ground_truth" in res
        assert "counter_en" in res
        assert "counter_hi" in res
