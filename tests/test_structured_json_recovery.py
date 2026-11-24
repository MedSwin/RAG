from app.services.prompts.structured import extract_json_object, schema_instruction


def test_extract_json_object_skips_schema_echo_and_reads_the_payload():
    text = (
        'Schema: {"type":"object","properties":{"claims":{"type":"array"}}}\n'
        'Here is the result: {"claims":[{"facet":"safety","claim":"avoid","chunk_id":"c1"}],"notes":[]}'
    )
    parsed = extract_json_object(text)
    assert parsed["claims"][0]["chunk_id"] == "c1"


def test_a_schema_echo_marks_the_specialist_degraded():
    import asyncio

    from app.agents.base import call_claim_agent
    from app.schemas.enums import SourceType
    from app.schemas.evidence import CandidatePassage

    class _Client:
        backend = "bedrock"

        def __init__(self, content: str):
            self.content = content

        async def call_llm(self, messages, json_schema=None, **kwargs):
            return {"content": self.content}

    passage = CandidatePassage(
        chunk_id="c1",
        doc_id="d1",
        source_type=SourceType.CPG,
        text="Guideline recommendation.",
    )
    echo = asyncio.run(
        call_claim_agent(
            _Client('{"type":"object","properties":{"claims":{"type":"array"}}}'),
            "safety",
            "Read the safety passage.",
            "Check the allergy.",
            [passage],
        )
    )
    assert echo.degraded is True
    assert echo.error == "schema_echo_only"
    assert echo.claims == []

    empty = asyncio.run(
        call_claim_agent(_Client("   "), "safety", "Read the safety passage.", "Check the allergy.", [passage])
    )
    assert empty.degraded is True
    assert empty.error == "schema_echo_only"

    real = asyncio.run(
        call_claim_agent(
            _Client('{"claims":[{"facet":"safety","claim":"Avoid","polarity":"safety","chunk_id":"c1","confidence":0.9}],"notes":[]}'),
            "safety",
            "Read the safety passage.",
            "Check the allergy.",
            [passage],
        )
    )
    assert real.degraded is False
    assert real.claims[0].chunk_id == "c1"

    unknown = asyncio.run(
        call_claim_agent(
            _Client('{"claims":[{"facet":"safety","claim":"Avoid","polarity":"safety","chunk_id":"missing","confidence":0.9}],"notes":[]}'),
            "safety",
            "Read the safety passage.",
            "Check the allergy.",
            [passage],
        )
    )
    assert unknown.degraded is True
    assert unknown.error == "uncited_claims"
    assert unknown.claims == []

    class _Broken(_Client):
        async def call_llm(self, messages, json_schema=None, **kwargs):
            raise RuntimeError("prompt text 555-234-5678")

    failed = asyncio.run(
        call_claim_agent(_Broken(""), "safety", "Read the safety passage.", "Check the allergy.", [passage])
    )
    assert failed.degraded is True
    assert failed.error == "RuntimeError"
    assert "555-234-5678" not in (failed.error or "")


def test_schema_instruction_adds_a_concrete_claim_example():
    instruction = schema_instruction({"type": "object", "properties": {"claims": {}, "notes": {}}})
    assert '"chunk_id":"id#0"' in instruction
    assert "Example:" in instruction
