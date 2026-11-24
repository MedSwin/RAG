"""Claim-level evidence ledger merge."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from app.core.config import settings
from app.schemas.agents import AgentClaimBatch
from app.schemas.enums import EvidencePolarity
from app.schemas.evidence import (
    CandidatePassage,
    ContradictionPair,
    EvidenceClaim,
    EvidenceLedgerEntry,
)
from app.schemas.facets import ClinicalFacet
from app.scoring.coverage import infer_polarity, score_passage_facets
from app.services.medswin.governance import lead_with_caution
from app.scoring.calibrate import clamp, first_present
from app.scoring.hierarchy import evidence_grade_from_metadata


def _provenance(passage: CandidatePassage) -> dict:
    return {
        "doc_id": passage.doc_id,
        "section": passage.section,
        "offset_start": passage.offset_start,
        "offset_end": passage.offset_end,
        "source_type": passage.source_type.value,
        "guideline_version": passage.metadata.get("guideline_version") or passage.metadata.get("version"),
        "effective_date": passage.metadata.get("effective_date"),
        "timestamp": passage.metadata.get("timestamp"),
    }


def _sentence_scopes(text: str) -> List[tuple]:
    """Split a note when the recommendation and the warning name different drugs."""
    from app.scoring.coverage import _QUALIFY, _ungoverned_caution

    caution_rows = []
    other_rows = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        sentence = sentence.strip()
        if not sentence:
            continue
        if _ungoverned_caution(sentence) or _QUALIFY.search(sentence):
            caution_rows.append(sentence)
        else:
            other_rows.append(sentence)
    caution = named_caution_text(" ".join(caution_rows)) if caution_rows else ""
    other = " ".join(other_rows)
    if intervention_tokens(caution) and intervention_tokens(other) and intervention_tokens(caution).isdisjoint(
        intervention_tokens(other)
    ):
        return [("support", other), ("caution", caution)]
    return [("", text or "")]


def build_retrieval_ledger(
    passages: List[CandidatePassage],
    facets: List[ClinicalFacet],
    agent_id: str = "retrieval",
) -> List[EvidenceLedgerEntry]:
    ledger: List[EvidenceLedgerEntry] = []
    for passage in passages:
        grade = evidence_grade_from_metadata(passage)
        scored = first_present(passage.calibrated_score, passage.rerank_score)
        if scored is None:
            # A rerank miss must not wipe facet coverage. Inclusion still requires calibrated_score.
            scored = first_present(passage.fusion_score, passage.dense_score, default=0.0)
        calibrated = clamp(scored)
        facet_scores = score_passage_facets(passage, facets)
        scopes = _sentence_scopes(passage.text or "")
        scope_scores = {"": facet_scores}
        if any(scope for scope, _scope_text in scopes):
            for scope, scope_text in scopes:
                scoped = passage.model_copy(update={"text": scope_text})
                scope_scores[scope] = score_passage_facets(scoped, facets)
        claims: List[EvidenceClaim] = []
        matched = []
        for facet in facets:
            for scope, scope_text in scopes:
                score = scope_scores.get(scope, facet_scores).get(facet.name, 0.0)
                if score <= 0.0:
                    continue
                if facet.name not in matched:
                    matched.append(facet.name)
                if scope == "caution":
                    from app.scoring.coverage import _QUALIFY, _ungoverned_caution

                    if _ungoverned_caution(scope_text):
                        polarity = (
                            EvidencePolarity.SAFETY
                            if facet.name == "safety_contraindications"
                            else EvidencePolarity.CONTRADICTS
                        )
                    elif _QUALIFY.search(scope_text):
                        polarity = EvidencePolarity.QUALIFIES
                    else:
                        polarity = EvidencePolarity.CONTRADICTS
                    claim_text = scope_text
                elif scope == "support":
                    from app.scoring.coverage import _QUALIFY

                    polarity = EvidencePolarity.QUALIFIES if _QUALIFY.search(scope_text) else EvidencePolarity.SUPPORTS
                    claim_text = scope_text
                else:
                    polarity = infer_polarity(passage, facet)
                    claim_text = " ".join(lead_with_caution(passage.text).split())[:280]
                provenance = _provenance(passage)
                if scope:
                    provenance = {**provenance, "sentence_scope": scope}
                claims.append(
                    EvidenceClaim(
                        facet=facet.name,
                        claim=" ".join(claim_text.split())[:280],
                        polarity=polarity,
                        chunk_id=passage.chunk_id,
                        confidence=clamp(score),
                        evidence_grade=grade,
                        calibrated_relevance=calibrated,
                        agent_id=agent_id,
                        provenance=provenance,
                    )
                )
        ledger.append(
            EvidenceLedgerEntry(
                chunk_id=passage.chunk_id,
                doc_id=passage.doc_id,
                source_type=passage.source_type,
                agent_id=agent_id,
                facets=matched,
                claims=claims,
                calibrated_relevance=calibrated,
                fusion_score=clamp(first_present(passage.fusion_score, default=calibrated)),
                evidence_grade=grade,
                safety_relevance=clamp(passage.safety_score or 0.0),
                contradiction_risk=clamp(passage.contradiction_score or 0.0),
                text=" ".join(passage.text.split()),
                provenance=_provenance(passage),
            )
        )
    return ledger


def merge_agent_claims(
    ledger: List[EvidenceLedgerEntry],
    batches: List[AgentClaimBatch],
    passages: List[CandidatePassage],
) -> List[EvidenceLedgerEntry]:
    by_chunk: Dict[str, EvidenceLedgerEntry] = {entry.chunk_id: entry for entry in ledger}
    passage_map = {p.chunk_id: p for p in passages}
    for batch in batches:
        for claim in batch.claims:
            passage = passage_map.get(claim.chunk_id)
            if passage is None:
                continue
            text_score = float((passage.facet_scores or {}).get(claim.facet, 0.0) or 0.0)
            if text_score <= 0.0:
                continue
            text_polarity = infer_polarity(passage, ClinicalFacet(name=claim.facet))
            updates: Dict[str, object] = {}
            caution_drugs = intervention_tokens(named_caution_text(passage.text or ""))
            claim_drugs = intervention_tokens(claim.claim or "")
            different_drug = bool(caution_drugs and claim_drugs and caution_drugs.isdisjoint(claim_drugs))
            if different_drug and claim.polarity == EvidencePolarity.SUPPORTS:
                # The caution sentence opened this facet. A recommendation for
                # another drug must not become the support that clears it.
                current = by_chunk.get(claim.chunk_id)
                has_support = any(
                    item.facet == claim.facet and item.polarity == EvidencePolarity.SUPPORTS
                    for item in (current.claims if current else [])
                )
                if not has_support:
                    continue
            if (
                text_polarity
                in {
                    EvidencePolarity.CONTRADICTS,
                    EvidencePolarity.SAFETY,
                    EvidencePolarity.QUALIFIES,
                }
                and claim.polarity == EvidencePolarity.SUPPORTS
                and not different_drug
            ):
                updates["polarity"] = text_polarity
            if text_score < 0.50 and float(claim.confidence or 0.0) > text_score:
                updates["confidence"] = text_score
            if updates:
                claim = claim.model_copy(update=updates)
            entry = by_chunk.get(claim.chunk_id)
            if entry is None:
                entry = EvidenceLedgerEntry(
                    chunk_id=passage.chunk_id,
                    doc_id=passage.doc_id,
                    source_type=passage.source_type,
                    agent_id=batch.agent_id,
                    provenance=_provenance(passage),
                    calibrated_relevance=clamp(passage.calibrated_score or 0.0),
                    fusion_score=clamp(passage.fusion_score or 0.0),
                    evidence_grade=evidence_grade_from_metadata(passage),
                    text=" ".join(passage.text.split()),
                )
                by_chunk[claim.chunk_id] = entry
                ledger.append(entry)
            entry.claims.append(claim)
            if claim.facet not in entry.facets:
                entry.facets.append(claim.facet)
            if passage is not None:
                passage.agent_scores[batch.agent_id] = max(
                    passage.agent_scores.get(batch.agent_id, 0.0),
                    float(claim.confidence or 0.0),
                )
                if batch.agent_id not in passage.retrieved_by:
                    passage.retrieved_by.append(batch.agent_id)
    return ledger


def filter_ledger(
    ledger: List[EvidenceLedgerEntry],
    chunk_ids: set,
) -> List[EvidenceLedgerEntry]:
    """Keep merged agent+retrieval claims for the selected bundle only."""
    return [entry.model_copy(deep=True) for entry in ledger if entry.chunk_id in chunk_ids]


def audit_chunk_ids(selected: List[CandidatePassage], ranked: List[CandidatePassage]) -> set:
    """Keep packed passages and included cautions for conflict detection."""
    from app.agents.base import _is_prompt_caution

    ids = {passage.chunk_id for passage in selected}
    for passage in ranked:
        if passage.chunk_id in ids:
            continue
        score = passage.calibrated_score
        if score is None or float(score) < settings.SUFF_T_INCLUSION:
            continue
        if _is_prompt_caution(passage):
            ids.add(passage.chunk_id)
    return ids


def cited_caution_passages(
    selected: List[CandidatePassage],
    ranked: List[CandidatePassage],
    contradictions: List[ContradictionPair],
) -> List[CandidatePassage]:
    """Return included cautions that a conflict cites and the bundle does not show."""
    from app.agents.base import _is_prompt_caution

    shown = {passage.chunk_id for passage in selected}
    cited = set()
    for pair in contradictions or []:
        if pair.resolved:
            continue
        cited.add(pair.chunk_id_a)
        cited.add(pair.chunk_id_b)
    extra: List[CandidatePassage] = []
    for passage in ranked:
        if passage.chunk_id not in cited or passage.chunk_id in shown:
            continue
        score = passage.calibrated_score
        if score is None or float(score) < settings.SUFF_T_INCLUSION:
            continue
        if not _is_prompt_caution(passage):
            continue
        extra.append(passage)
        shown.add(passage.chunk_id)
    return extra


def apply_claim_alignment(
    passages: List[CandidatePassage],
    batches: List[AgentClaimBatch],
    facets: List[ClinicalFacet],
) -> None:
    """Keep the text score. A specialist confidence does not raise it."""
    del passages, batches, facets


_STALE_TERMS = ("outdated", "obsolete", "superseded", "population mismatch")


def adjudicate_contradictions(
    pairs: List[ContradictionPair],
    batches: List[AgentClaimBatch],
) -> List[ContradictionPair]:
    """Resolve one medium conflict only when the critic cites both chunks."""
    cited_stale: set[str] = set()
    for batch in batches:
        if batch.agent_id != "critic":
            continue
        for claim in batch.claims:
            text = f"{claim.claim} {claim.chunk_id}".lower()
            if claim.chunk_id and any(term in text for term in _STALE_TERMS):
                cited_stale.add(claim.chunk_id)
    for pair in pairs:
        if pair.resolved or pair.severity == "high":
            continue
        if pair.chunk_id_a in cited_stale and pair.chunk_id_b in cited_stale:
            pair.resolved = True
            pair.adjudication = "critic cited both chunks as outdated or mismatched"
    return pairs


def _contradicts_facet(claim: EvidenceClaim, facet: str) -> bool:
    if claim.polarity == EvidencePolarity.CONTRADICTS:
        return True
    # A safety warning on a treatment/guideline facet is a conflict, not support.
    return claim.polarity == EvidencePolarity.SAFETY and facet != "safety_contraindications"


def _supports_facet(claim: EvidenceClaim, facet: str) -> bool:
    if claim.polarity == EvidencePolarity.IRRELEVANT:
        return False
    if _contradicts_facet(claim, facet):
        return False
    # A safety warning covers the safety facet. It is not a recommendation.
    return claim.polarity == EvidencePolarity.SUPPORTS


_DRUG_WORDS = {
    "apixaban",
    "dabigatran",
    "rivaroxaban",
    "edoxaban",
    "warfarin",
    "heparin",
    "enoxaparin",
    "aspirin",
    "nsaid",
    "nsaids",
    "ibuprofen",
    "ketorolac",
    "metformin",
}
_SHORT_ALIASES = {
    "asa": "aspirin",
}
_DRUG_ALIASES = {
    "eliquis": "apixaban",
    "pradaxa": "dabigatran",
    "xarelto": "rivaroxaban",
    "savaysa": "edoxaban",
    "lixiana": "edoxaban",
    "coumadin": "warfarin",
    "clexane": "enoxaparin",
    "lovenox": "enoxaparin",
    "glucophage": "metformin",
}
_DRUG_SUFFIXES = (
    "xaban",
    "gatran",
    "farin",
    "parin",
    "profen",
    "cillin",
    "mycin",
    "statin",
    "sartan",
    "pril",
    "olol",
    "azole",
    "mab",
    "nib",
)
_WORD = re.compile(r"[a-z][a-z0-9-]{3,}")
_NOT_DRUGS = {"april"}
_DOSE_IN_SENTENCE = re.compile(
    r"\b(?:dose|dosing|mg|tablet|tab|capsule|daily|\d+(?:\.\d+)?(?:mg|mcg|g|ml|units?))\b",
    re.IGNORECASE,
)
_DOSE_TOKEN = re.compile(
    r"^(?:dose|dosing|mg|mcg|tablet|tab|capsule|daily|\d+(?:\.\d+)?(?:mg|mcg|g|ml|units?))$",
    re.IGNORECASE,
)
_ALIAS_CONTEXT_STOP = re.compile(r"\b(?:meeting|conference|society)\b", re.IGNORECASE)


def dose_span(sentence: str) -> str:
    """Keep a drug and its dose. History words around that dose stay out."""
    if _ALIAS_CONTEXT_STOP.search(sentence or ""):
        return ""
    if not _DOSE_IN_SENTENCE.search(sentence or ""):
        return ""
    words = (sentence or "").split()
    dose_at = []
    drug_at = []
    for index, word in enumerate(words):
        token = word.lower().strip(".,;")
        if _DOSE_TOKEN.fullmatch(token):
            dose_at.append(index)
        if token in _SHORT_ALIASES or intervention_tokens(word):
            drug_at.append(index)
    windows = []
    for dose_index in dose_at:
        nearby = [index for index in drug_at if dose_index - 4 <= index <= dose_index + 4]
        if not nearby:
            continue
        begin = min([dose_index, *nearby])
        end = max([dose_index, *nearby]) + 1
        while end < len(words) and end <= dose_index + 3:
            nxt = words[end].lower().strip(".,;")
            if _DOSE_TOKEN.fullmatch(nxt):
                end += 1
            else:
                break
        windows.append((begin, end))
    if not windows:
        return ""
    merged = []
    for begin, end in windows:
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((begin, end))
    return " ".join(" ".join(words[begin:end]) for begin, end in merged)


def caution_alias_tokens(text: str) -> set[str]:
    """Map a short drug abbreviation in a caution or a dose line."""
    from app.scoring.coverage import _QUALIFY, _ungoverned_caution

    found: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        lowered = sentence.lower()
        caution = _ungoverned_caution(sentence) or bool(_QUALIFY.search(sentence))
        dose = bool(_DOSE_IN_SENTENCE.search(sentence)) and not _ALIAS_CONTEXT_STOP.search(sentence)
        if not caution and not dose:
            continue
        for alias, generic in _SHORT_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", lowered):
                found.add(generic)
    return found


def intervention_tokens(text: str) -> set[str]:
    """Return named drugs in one passage. Class words such as anticoagulant stay out."""
    found: set[str] = caution_alias_tokens(text)
    token = (text or "").strip().lower().strip(".,;")
    if token in _SHORT_ALIASES:
        found.add(_SHORT_ALIASES[token])
    for raw in _WORD.findall((text or "").lower()):
        for word in raw.split("-"):
            if len(word) < 4 or word in _NOT_DRUGS:
                continue
            if word.endswith("s") and word[:-1] in _DRUG_WORDS:
                word = word[:-1]
            alias = _DRUG_ALIASES.get(word)
            if alias:
                found.add(alias)
                continue
            listed = next((drug for drug in _DRUG_WORDS if len(drug) >= 6 and word.startswith(drug)), None)
            if listed:
                found.add(listed)
            elif word in _DRUG_WORDS or word.endswith(_DRUG_SUFFIXES):
                found.add(word)
    return found


def same_intervention(left: str, right: str) -> bool:
    """Pair a caution with support only when both name the same drug, or neither names one.

    A named-drug warning does not contradict a passage that names a different drug
    or no drug. The warning still covers the safety facet.
    """
    left_drugs = intervention_tokens(left)
    right_drugs = intervention_tokens(right)
    if left_drugs and right_drugs:
        return not left_drugs.isdisjoint(right_drugs)
    if left_drugs or right_drugs:
        return False
    return True


def _pair_severity(left: EvidenceLedgerEntry, right: EvidenceLedgerEntry) -> str:
    return (
        "high"
        if left.evidence_grade.score >= 0.80 or right.evidence_grade.score >= 0.80
        else "medium"
    )


def _reason_with_drug(reason: str, left: str, right: str) -> str:
    """Name the shared drug in the ledger line the clinician reads."""
    drugs = sorted(intervention_tokens(left) & intervention_tokens(right))
    if not drugs:
        return reason
    return f"{reason.rstrip('.')} for {', '.join(drugs)}."


def _pairing_text(entry: EvidenceLedgerEntry, claim: Optional[EvidenceClaim] = None) -> str:
    """Use the claim when it names a drug. A cut preview still uses the passage."""
    if claim is not None and (claim.provenance or {}).get("sentence_scope") and (claim.claim or "").strip():
        return claim.claim.strip()
    claim_text = (claim.claim or "").strip() if claim is not None else ""
    stored = (entry.text or "").strip()
    if claim_text and intervention_tokens(claim_text):
        return claim_text
    if stored:
        return stored
    if claim_text:
        return claim_text
    return " ".join(item.claim for item in entry.claims)


def named_caution_text(text: str) -> str:
    """Return the caution sentences when they name a drug."""
    from app.retrieval.lexical import LexicalRetriever
    from app.scoring.coverage import _QUALIFY, _ungoverned_caution

    spans = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        if not sentence.strip():
            continue
        if _ungoverned_caution(sentence) or _QUALIFY.search(sentence):
            span = LexicalRetriever._caution_span(sentence).strip()
            if span:
                spans.append(span)
    caution = " ".join(spans)
    if caution and intervention_tokens(caution):
        return caution
    return text or ""


def _conflict_text(entry: EvidenceLedgerEntry, claim: Optional[EvidenceClaim] = None) -> str:
    """Use the caution sentence when it names the drug. A support sentence stays out."""
    return named_caution_text(_pairing_text(entry, claim))


def _support_text(entry: EvidenceLedgerEntry, claim: Optional[EvidenceClaim] = None) -> str:
    """Drop a caution sentence that names a different drug from the recommendation."""
    from app.scoring.coverage import _QUALIFY, _ungoverned_caution

    text = _pairing_text(entry, claim)
    caution: List[str] = []
    other: List[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        if not sentence.strip():
            continue
        if _ungoverned_caution(sentence) or _QUALIFY.search(sentence):
            caution.append(sentence.strip())
        else:
            other.append(sentence.strip())
    other_text = " ".join(other)
    caution_text = " ".join(caution)
    other_drugs = intervention_tokens(other_text)
    caution_drugs = intervention_tokens(caution_text)
    if other_drugs and caution_drugs and other_drugs.isdisjoint(caution_drugs):
        return other_text
    return text


PATIENT_CHART_ID = "patient-chart"


def _chart_caution_text(chart: str) -> str:
    """Keep caution sentences from a chart note. A later drug sentence stays out."""
    from app.retrieval.lexical import LexicalRetriever
    from app.scoring.coverage import _QUALIFY, _ungoverned_caution

    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", chart or ""):
        if not sentence.strip():
            continue
        if _ungoverned_caution(sentence) or _QUALIFY.search(sentence):
            kept.append(LexicalRetriever._caution_span(sentence))
    return " ".join(kept).strip()


def chart_contradictions(chart: str, ledger: List[EvidenceLedgerEntry]) -> List[ContradictionPair]:
    """Pair a chart caution with a selected recommendation for the same drug.

    The chart synopsis is not a retrieved passage. A dose line is not a caution.
    A drug named outside the caution sentence does not form a conflict.
    """
    from app.scoring.coverage import _QUALIFY, _ungoverned_caution

    text = _chart_caution_text(chart)
    if not text or not intervention_tokens(text):
        return []
    caution = _ungoverned_caution(text)
    qualify = bool(_QUALIFY.search(text)) and not caution
    if not caution and not qualify:
        return []
    pairs: List[ContradictionPair] = []
    seen = set()
    for entry in ledger:
        if entry.chunk_id == PATIENT_CHART_ID:
            continue
        for claim in entry.claims:
            if claim.polarity != EvidencePolarity.SUPPORTS or claim.facet == "safety_contraindications":
                continue
            support_text = _support_text(entry, claim)
            if not intervention_tokens(support_text) or not same_intervention(support_text, text):
                continue
            key = (claim.facet, entry.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            if caution:
                reason = "Same-drug safety warning conflicts with this recommendation."
                severity = "high" if entry.evidence_grade.score >= 0.80 else "medium"
            else:
                reason = "Same-drug caution qualifies this recommendation."
                severity = "low"
            span = " ".join(text.split())
            pairs.append(
                ContradictionPair(
                    facet=claim.facet,
                    chunk_id_a=entry.chunk_id,
                    chunk_id_b=PATIENT_CHART_ID,
                    severity=severity,
                    reason=(
                        f"Chart caution: {span} | "
                        f"{_reason_with_drug(reason, support_text, text)}"
                    ),
                    grade_a=entry.evidence_grade.score,
                    grade_b=0.0,
                )
            )
    return pairs


def detect_contradictions(ledger: List[EvidenceLedgerEntry]) -> List[ContradictionPair]:
    by_facet: Dict[str, Dict[str, List[EvidenceLedgerEntry]]] = {}
    for entry in ledger:
        for claim in entry.claims:
            bucket = by_facet.setdefault(claim.facet, {"support": [], "contradict": []})
            if _contradicts_facet(claim, claim.facet):
                bucket["contradict"].append(entry)
            elif _supports_facet(claim, claim.facet):
                bucket["support"].append(entry)
    pairs: List[ContradictionPair] = []
    seen = set()

    def _add(
        facet: str,
        support: EvidenceLedgerEntry,
        conflict: EvidenceLedgerEntry,
        reason: str,
        severity: str | None = None,
    ) -> None:
        key = (facet, support.chunk_id, conflict.chunk_id)
        if support.chunk_id == conflict.chunk_id or key in seen:
            return
        seen.add(key)
        pairs.append(
            ContradictionPair(
                facet=facet,
                chunk_id_a=support.chunk_id,
                chunk_id_b=conflict.chunk_id,
                severity=severity or _pair_severity(support, conflict),
                reason=reason,
                grade_a=support.evidence_grade.score,
                grade_b=conflict.evidence_grade.score,
            )
        )

    for facet, bucket in by_facet.items():
        supports = list({id(entry): entry for entry in bucket["support"]}.values())
        conflicts = list({id(entry): entry for entry in bucket["contradict"]}.values())
        for support in supports:
            support_claims = [
                claim for claim in support.claims if claim.facet == facet and _supports_facet(claim, facet)
            ]
            for conflict in conflicts:
                conflict_claims = [
                    claim for claim in conflict.claims if claim.facet == facet and _contradicts_facet(claim, facet)
                ]
                for support_claim in support_claims:
                    for conflict_claim in conflict_claims:
                        support_text = _support_text(support, support_claim)
                        conflict_text = _conflict_text(conflict, conflict_claim)
                        if not same_intervention(support_text, conflict_text):
                            continue
                        _add(
                            facet,
                            support,
                            conflict,
                            _reason_with_drug(
                                "Incompatible support and caution for the same intervention.",
                                support_text,
                                conflict_text,
                            ),
                        )

    safety_warnings: List[tuple[EvidenceLedgerEntry, EvidenceClaim]] = []
    other_supports: List[tuple[EvidenceLedgerEntry, EvidenceClaim]] = []
    for entry in ledger:
        for claim in entry.claims:
            if claim.facet == "safety_contraindications" and claim.polarity in {
                EvidencePolarity.SAFETY,
                EvidencePolarity.CONTRADICTS,
            }:
                safety_warnings.append((entry, claim))
            elif (
                claim.polarity == EvidencePolarity.SUPPORTS
                and claim.facet != "safety_contraindications"
            ):
                other_supports.append((entry, claim))
    for support_entry, support_claim in other_supports:
        for warning_entry, warning_claim in safety_warnings:
            # A safety warning crosses facets only when both sides name the same drug.
            # An unnamed caution still covers the safety facet. It does not veto every guideline.
            if not intervention_tokens(_support_text(support_entry, support_claim)) or not intervention_tokens(
                _conflict_text(warning_entry, warning_claim)
            ):
                continue
            if not same_intervention(
                _support_text(support_entry, support_claim),
                _conflict_text(warning_entry, warning_claim),
            ):
                continue
            _add(
                support_claim.facet,
                support_entry,
                warning_entry,
                _reason_with_drug(
                    "Same-drug safety warning conflicts with this recommendation.",
                    _support_text(support_entry, support_claim),
                    _conflict_text(warning_entry, warning_claim),
                ),
            )
    qualifies: List[tuple[EvidenceLedgerEntry, EvidenceClaim]] = []
    plain_supports: List[tuple[EvidenceLedgerEntry, EvidenceClaim]] = []
    for entry in ledger:
        for claim in entry.claims:
            if claim.polarity == EvidencePolarity.QUALIFIES:
                qualifies.append((entry, claim))
            elif claim.polarity == EvidencePolarity.SUPPORTS:
                plain_supports.append((entry, claim))
    for support_entry, support_claim in plain_supports:
        for qualify_entry, qualify_claim in qualifies:
            if not intervention_tokens(_support_text(support_entry, support_claim)) or not intervention_tokens(
                _conflict_text(qualify_entry, qualify_claim)
            ):
                continue
            if not same_intervention(
                _support_text(support_entry, support_claim),
                _conflict_text(qualify_entry, qualify_claim),
            ):
                continue
            _add(
                support_claim.facet,
                support_entry,
                qualify_entry,
                _reason_with_drug(
                    "Same-drug caution qualifies this recommendation.",
                    _support_text(support_entry, support_claim),
                    _conflict_text(qualify_entry, qualify_claim),
                ),
                severity="low",
            )
    return pairs
