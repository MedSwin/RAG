from app.services.prompts.structured import extract_json_object, schema_instruction


def test_extract_json_object_skips_schema_echo_and_reads_the_payload():
    text = (
        'Schema: {"type":"object","properties":{"claims":{"type":"array"}}}\n'
        'Here is the result: {"claims":[{"facet":"safety","claim":"avoid","chunk_id":"c1"}],"notes":[]}'
    )
    parsed = extract_json_object(text)
    assert parsed["claims"][0]["chunk_id"] == "c1"


def test_schema_instruction_adds_a_concrete_claim_example():
    instruction = schema_instruction({"type": "object", "properties": {"claims": {}, "notes": {}}})
    assert '"chunk_id":"id#0"' in instruction
    assert "Example:" in instruction
