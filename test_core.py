"""
Unit and integration tests for CivicTrustHub rumor detector pipeline.
Tests rumor cue detection, forward stripping, dedicated fact-check desk matching,
multi-source confirmation rule, temporal guards, and Unicode preservation.
"""
from unittest.mock import patch
import main

# 1. Rumor Cue Profiling Tests
def test_detect_forward_urgency_cues():
    text = "URGENT!! Forward to all groups immediately! Cyclone alert issued!"
    cues = main.detect_rumor_cues(text)
    flag_codes = [f["code"] for f in cues["flags"]]
    assert "FORWARD_URGENCY" in flag_codes
    assert cues["is_suspicious_forward"] or cues["cue_count"] >= 1

def test_detect_sensational_and_vague_authority():
    text = "SHOCKING breaking news: Doctors say secret cure found for all viruses! 100% true"
    cues = main.detect_rumor_cues(text)
    flag_codes = [f["code"] for f in cues["flags"]]
    assert "SENSATIONALISM" in flag_codes
    assert "VAGUE_AUTHORITY" in flag_codes
    assert cues["is_suspicious_forward"]

def test_detect_all_caps_flag():
    text = "DELHI AIRPORT COMPLETELY SUBMERGED UNDER WATER EMERGENCY"
    cues = main.detect_rumor_cues(text)
    flag_codes = [f["code"] for f in cues["flags"]]
    assert "ALL_CAPS" in flag_codes

# 2. Forward Boilerplate Stripping & Claim Extraction
def test_extract_rumor_claim_from_boilerplate():
    raw = "URGENT!! Share with all: Delhi airport runway flooded and closed"
    clean = main.extract_rumor_claim(raw)
    assert clean.lower().startswith("delhi airport runway")
    assert "urgent" not in clean.lower()

# 3. Linguistic & Name Preservation Tests
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

# 4. Strict Unicode & Indic Script Preservation
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

# 5. Conversational Chitchat & Typo Filtering
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

# 6. Dedicated Fact-Check Sweep & Debunk Detection
def test_fact_check_desk_match_flags_rumor_debunked():
    mock_fc = [{
        "source": "BOOM Live",
        "text": "Fact Check: No, Delhi airport runway is not submerged; old Dubai video shared as recent"
    }]
    with patch("main.fetch_rumor_evidence", return_value=([], mock_fc, None)):
        res = main.verify_claim_against_web("Delhi airport runway flooded")
        assert res["verdict"] == "RUMOR (debunked)"
        assert "BOOM Live" in res["ground_truth"]["source"]

# 7. Absence of Evidence on Viral Claims
def test_absence_of_coverage_on_viral_forward_flags_likely_rumor():
    with patch("main.fetch_rumor_evidence", return_value=([], [], None)):
        res = main.verify_claim_against_web("URGENT Forward to all: Famous celebrity died in Mumbai shooting")
        assert res["verdict"] == "LIKELY RUMOR"
        assert res["cues"]["cue_count"] >= 1

# 8. Temporal Guard (Future Release Asserted as Past)
def test_future_release_flagged_as_debunked_rumor():
    mock_news = [{
        "source": "Film Industry Wire",
        "text": "The sequel is scheduled for release in December 2028."
    }]
    with patch("main.fetch_rumor_evidence", return_value=(mock_news, [], None)):
        res = main.verify_claim_against_web("Sequel released")
        assert res["verdict"] == "RUMOR (debunked)"
        assert "Future" in res["classification"]

# 9. Multi-Source Corroboration Rule
def test_multi_source_required_for_confirmed():
    mock_news_two_sources = [
        {"source": "Press Trust of India", "text": "IMD issues red alert for severe rainfall in coastal districts"},
        {"source": "The Hindu", "text": "Weather bureau warns of severe coastal rainfall this week"}
    ]
    with patch("main.fetch_rumor_evidence", return_value=(mock_news_two_sources, [], None)), \
         patch("main.evaluate_nli_entailment", return_value=[
             {"entailment": 0.85, "contradiction": 0.05, "neutral": 0.10},
             {"entailment": 0.82, "contradiction": 0.03, "neutral": 0.15}
         ]):
        res = main.verify_claim_against_web("Heavy rainfall warning issued for coastal areas")
        assert res["verdict"] == "CONFIRMED"
        assert res["sources_count"] >= 2

def test_single_source_remains_unconfirmed():
    mock_news_single = [
        {"source": "Local Blog", "text": "Heavy rainfall warning issued for coastal areas"}
    ]
    with patch("main.fetch_rumor_evidence", return_value=(mock_news_single, [], None)), \
         patch("main.evaluate_nli_entailment", return_value=[
             {"entailment": 0.85, "contradiction": 0.05, "neutral": 0.10}
         ]):
        res = main.verify_claim_against_web("Heavy rainfall warning issued for coastal areas")
        assert res["verdict"] == "UNCONFIRMED"
        assert res["sources_count"] == 1

# 10. Network Failure Resilience
def test_network_timeout_reported_as_unavailable():
    with patch("main.fetch_rumor_evidence", return_value=([], [], "Connection timed out after 7000ms")):
        res = main.verify_claim_against_web("Cyclone warning issued for Odisha")
        assert res["verdict"] == "NETWORK_ERROR"
        assert res["classification"] == "SERVICE TEMPORARILY UNAVAILABLE"
