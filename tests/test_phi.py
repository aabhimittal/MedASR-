from medasr.phi import contains_phi, find_phi, redact


def kinds(text):
    return {m.kind for m in find_phi(text)}


def test_redacts_phone():
    r = redact("call me at 555-123-4567 please")
    assert "555-123-4567" not in r.redacted_text
    assert "[PHONE]" in r.redacted_text


def test_phone_formats():
    for p in ["(555) 123-4567", "555.123.4567", "+1 555 123 4567", "5551234567"]:
        assert "PHONE" in kinds(f"reach {p} now"), p


def test_redacts_ssn():
    r = redact("ssn 123-45-6789")
    assert "[SSN]" in r.redacted_text and "123-45-6789" not in r.redacted_text


def test_redacts_email_and_url():
    r = redact("mail dr.smith@hospital.org or see https://portal.example.com/x")
    assert "[EMAIL]" in r.redacted_text
    assert "[URL]" in r.redacted_text
    assert "@hospital.org" not in r.redacted_text


def test_redacts_mrn_with_cue_word():
    r = redact("patient MRN 4456721 admitted")
    assert "[MRN]" in r.redacted_text
    assert "4456721" not in r.redacted_text
    # The cue word itself is informative and stays.
    assert "MRN" in r.redacted_text


def test_mrn_variants():
    for cue in ["MRN:", "medical record number", "record number", "mrn #"]:
        assert "MRN" in kinds(f"{cue} 1234567"), cue


def test_redacts_dates():
    for d in ["03/14/2024", "3-14-24", "March 14, 2024"]:
        assert "DATE" in kinds(f"seen on {d}"), d


def test_redacts_age_over_89():
    assert "AGE_OVER_89" in kinds("a 92-year-old male")
    assert "AGE_OVER_89" in kinds("a 104 year old female")


def test_age_under_90_not_flagged():
    assert "AGE_OVER_89" not in kinds("a 45-year-old male")


def test_redacts_zip():
    assert "ZIP" in kinds("lives in 02139")


def test_clean_clinical_text_untouched():
    text = "patient denies chest pain and reports mild headache"
    r = redact(text)
    assert r.redacted_text == text
    assert not r.found_phi


def test_dosage_not_mistaken_for_phi():
    """A dose must never be redacted -- that would corrupt the prescription."""
    text = "start metformin 500 mg twice daily"
    r = redact(text)
    assert r.redacted_text == text
    assert not r.found_phi


def test_counts_are_reported():
    r = redact("call 555-123-4567 or 555-987-6543, mail a@b.co")
    assert r.counts.get("PHONE") == 2
    assert r.counts.get("EMAIL") == 1


def test_overlapping_matches_resolved_once():
    """A span must be claimed by exactly one pattern, never double-redacted."""
    r = redact("ssn 123-45-6789")
    assert r.redacted_text.count("[") == 1


def test_multiple_redactions_keep_offsets_valid():
    text = "on 03/14/2024 call 555-123-4567 mrn 8899001 zip 02139"
    r = redact(text)
    for m in r.matches:
        assert text[m.start:m.end] == m.text  # offsets point at the real span
    assert "[DATE]" in r.redacted_text and "[PHONE]" in r.redacted_text


def test_contains_phi_helper():
    assert contains_phi("call 555-123-4567")
    assert not contains_phi("no known drug allergies")


def test_custom_placeholder():
    r = redact("call 555-123-4567", placeholder={"PHONE": "<<TEL>>"})
    assert "<<TEL>>" in r.redacted_text


def test_empty_string():
    r = redact("")
    assert r.redacted_text == "" and not r.found_phi
