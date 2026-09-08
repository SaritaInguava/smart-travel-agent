from smart_travel_agent.guardrails import check_confidence_calibration


def test_well_calibrated_text_passes_with_no_flags():
    text = (
        "The region is generally considered safe for tourists, though standard urban "
        "precautions apply. Most visitors will not need a visa for short stays."
    )
    result = check_confidence_calibration(text)

    assert result["verdict"] == "PASS"
    assert result["flagged_phrases"] == []
    assert result["hedged_text"] == text


def test_overconfident_text_is_flagged_and_hedged():
    text = "This neighborhood is completely safe. It is guaranteed you will have zero risk."
    result = check_confidence_calibration(text)

    assert result["verdict"] == "FAIL"
    assert result["overconfidence_score"] >= 0.5
    flagged_markers = {m for m, _, _ in result["flagged_phrases"]}
    assert "completely safe" in flagged_markers
    assert "guaranteed" in flagged_markers
    assert "zero risk" in flagged_markers
    assert "completely safe" not in result["hedged_text"].lower()
    assert "guaranteed" not in result["hedged_text"].lower()


def test_hedging_preserves_unrelated_text():
    text = "You will always need to check current visa requirements before booking."
    result = check_confidence_calibration(text)

    assert "typically" in result["hedged_text"]
    assert "before booking" in result["hedged_text"]
