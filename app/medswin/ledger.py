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


def _is_included_order(passage: CandidatePassage) -> bool:
    """True when the passage orders a drug and can conflict with a warning."""
    text = passage.text or ""
    recommendation = _chart_recommendation_text(text)
    continuation = _chart_continuation_text(text)
    return bool(
        (recommendation and intervention_tokens(recommendation))
        or (continuation and intervention_tokens(continuation))
    )


def audit_chunk_ids(selected: List[CandidatePassage], ranked: List[CandidatePassage]) -> set:
    """Keep packed passages, included cautions, and included orders."""
    from app.agents.base import _is_prompt_caution

    ids = {passage.chunk_id for passage in selected}
    for passage in ranked:
        if passage.chunk_id in ids:
            continue
        score = passage.calibrated_score
        if score is None or float(score) < settings.SUFF_T_INCLUSION:
            continue
        if _is_prompt_caution(passage) or _is_included_order(passage):
            ids.add(passage.chunk_id)
    return ids


def cited_caution_passages(
    selected: List[CandidatePassage],
    ranked: List[CandidatePassage],
    contradictions: List[ContradictionPair],
) -> List[CandidatePassage]:
    """Return included cautions and orders that a conflict cites and the bundle does not show."""
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
        if not _is_prompt_caution(passage) and not _is_included_order(passage):
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
    from app.scoring.coverage import (
        _QUALIFY,
        _not_brought_stop,
        _not_home_stop,
        _not_own_stop,
        _not_current_stop,
        _not_regular_med_stop,
        _ungoverned_caution,
    )

    spans = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        if not sentence.strip():
            continue
        if _ungoverned_caution(sentence) or _QUALIFY.search(sentence):
            if (
                _not_regular_med_stop(sentence)
                or _not_home_stop(sentence)
                or _not_brought_stop(sentence)
                or _not_own_stop(sentence)
                or _not_current_stop(sentence)
            ):
                spans.append(sentence.strip())
                continue
            recommendation, clause = _caution_clause(sentence)
            recommendation_drugs = intervention_tokens(recommendation)
            clause_drugs = intervention_tokens(clause)
            order = _chart_recommendation_text(recommendation) or _chart_continuation_text(recommendation)
            disjoint = bool(
                recommendation_drugs
                and clause_drugs
                and recommendation_drugs.isdisjoint(clause_drugs)
            )
            if order and disjoint and _ungoverned_caution(clause):
                span = clause
            elif disjoint and _QUALIFY.search(sentence) and not _ungoverned_caution(clause):
                span = order or recommendation
            elif order and disjoint:
                span = clause
            elif clause_drugs and _ungoverned_caution(clause) and not recommendation_drugs:
                span = clause
            else:
                span = LexicalRetriever._caution_span(sentence).strip()
            if span:
                spans.append(span)
    caution = " ".join(spans)
    if caution and intervention_tokens(caution):
        return caution
    return text or ""


def _conflict_text(entry: EvidenceLedgerEntry, claim: Optional[EvidenceClaim] = None) -> str:
    """Use the caution sentence when it names the drug. A support sentence stays out."""
    paired = _pairing_text(entry, claim)
    stored = " ".join((entry.text or "").split())
    if stored and stored != paired and stored in paired:
        paired = stored
    return named_caution_text(paired)


def _order_boundary_word(word: str) -> bool:
    """True when this word starts an order for a later drug."""
    token = (word or "").strip(".,;:").lower()
    if token in {"use", "give", "continue", "continuing", "resume", "restart", "recommence"}:
        return True
    return bool(_RECOMMEND_VERB.fullmatch(token) or _START_VERB.fullmatch(token))


def _caution_clause(sentence: str) -> tuple[str, str]:
    """Split one sentence into the recommendation and a caution about another drug."""
    from app.scoring.coverage import (
        _BENIGN_DO_NOT,
        _CAUTION,
        _LOCAL_NEGATION,
        _QUALIFY,
        _STOP_ORDER,
        _COME_OFF,
        _NO_LONGER,
        _OFF_DRUG,
        _NOT_FOR,
        _NOT_ON,
        _REFUSED,
        _away_split,
        _negated_away_split,
    )

    negated = _negated_away_split(sentence)
    if negated:
        return negated
    away = _away_split(sentence)
    if away:
        return away

    hard_matches = []
    soft_matches = []
    for item in list(_CAUTION.finditer(sentence or "")) + list(_QUALIFY.finditer(sentence or "")) + list(
        _STOP_ORDER.finditer(sentence or "")
    ) + list(_COME_OFF.finditer(sentence or "")) + list(_NO_LONGER.finditer(sentence or "")) + list(
        _OFF_DRUG.finditer(sentence or "")
    ) + list(_NOT_FOR.finditer(sentence or "")) + list(_NOT_ON.finditer(sentence or "")) + list(
        _REFUSED.finditer(sentence or "")
    ):
        window = sentence[max(0, item.start() - 48):item.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        phrase = item.group(0).lower()
        if phrase in {"do not", "should not"} and _BENIGN_DO_NOT.match(sentence[item.end():]):
            continue
        if _CAUTION.fullmatch(item.group(0)) or _COME_OFF.fullmatch(item.group(0)):
            hard_matches.append(item)
        else:
            soft_matches.append(item)
    candidates = hard_matches or soft_matches
    match = min(candidates, key=lambda item: item.start()) if candidates else None
    if match is None:
        return sentence, ""
    words = (sentence or "").split()
    cursor = 0
    spans = []
    for word in words:
        at = sentence.find(word, cursor)
        spans.append(at)
        cursor = at + len(word)
    anchor = 0
    match_end = 0
    for index, begin in enumerate(spans):
        end = begin + len(words[index])
        if begin < match.end() and end > match.start():
            match_end = index
        if begin <= match.start() < end:
            anchor = index
    drug_at = [index for index, word in enumerate(words) if intervention_tokens(word)]
    before = [index for index in drug_at if index < anchor]
    after = [index for index in drug_at if index > anchor]
    before_drugs: set[str] = set()
    after_drugs: set[str] = set()
    for index in before:
        before_drugs.update(intervention_tokens(words[index]))
    for index in after:
        after_drugs.update(intervention_tokens(words[index]))
    after_near = [index for index in after if index <= anchor + 4]
    hard_caution = bool(_CAUTION.fullmatch(match.group(0)))
    nearby = [index for index in drug_at if anchor - 4 <= index <= anchor + 4]
    if (
        hard_caution
        and before_drugs
        and after_drugs
        and before_drugs.isdisjoint(after_drugs)
        and after_near
    ):
        after_text = " ".join(words[match_end + 1:])
        later_order = _has_recommendation_verb(after_text) and set(
            intervention_tokens(after_text)
        ).isdisjoint(before_drugs)
        if later_order:
            clause_begin = before[-1]
            clause_end = match_end + 1
        else:
            clause_begin = anchor
            clause_end = max([anchor, *after_near]) + 1
    elif (
        not hard_caution
        and before_drugs
        and after_drugs
        and before_drugs.isdisjoint(after_drugs)
        and after_near
    ):
        clause_begin = after_near[0]
        clause_end = max(after_near) + 1
    elif (
        _STOP_ORDER.fullmatch(match.group(0) or "")
        or _COME_OFF.fullmatch(match.group(0) or "")
        or _NO_LONGER.fullmatch(match.group(0) or "")
        or _OFF_DRUG.fullmatch(match.group(0) or "")
        or _NOT_FOR.fullmatch(match.group(0) or "")
        or _NOT_ON.fullmatch(match.group(0) or "")
        or _REFUSED.fullmatch(match.group(0) or "")
    ) and not before and after:
        limit = min(len(words), anchor + 9)
        split_at = None
        for index in range(anchor + 1, limit):
            earlier_object = any(anchor < drug_index < index for drug_index in drug_at)
            later_drug = any(index < drug_index < limit for drug_index in drug_at)
            if earlier_object and later_drug and _order_boundary_word(words[index]):
                split_at = index
                break
        objects = [
            drug_index
            for drug_index in drug_at
            if anchor < drug_index < (split_at if split_at is not None else limit)
        ]
        if objects:
            clause_begin = anchor
            clause_end = objects[-1] + 1
        elif nearby:
            clause_begin = min([anchor, *nearby])
            clause_end = max([anchor, *nearby]) + 1
        else:
            clause_begin = anchor
            clause_end = min(len(words), anchor + 3)
    elif hard_caution and not after and len(before) >= 2:
        nearest = before[-1]
        nearest_drugs = set(intervention_tokens(words[nearest]))
        earlier_drugs = before_drugs - nearest_drugs
        earlier = " ".join(words[:nearest])
        if (
            earlier_drugs
            and nearest_drugs
            and earlier_drugs.isdisjoint(nearest_drugs)
            and _has_recommendation_verb(earlier)
        ):
            clause_begin = nearest
            clause_end = max(match_end, nearest) + 1
        elif nearby:
            clause_begin = min([anchor, *nearby])
            clause_end = max([anchor, *nearby]) + 1
        else:
            clause_begin = anchor
            clause_end = min(len(words), anchor + 3)
    elif nearby:
        clause_begin = min([anchor, *nearby])
        clause_end = max([anchor, *nearby]) + 1
    else:
        clause_begin = anchor
        clause_end = min(len(words), anchor + 3)
    recommendation = " ".join(words[:clause_begin] + words[clause_end:])
    clause = " ".join(words[clause_begin:clause_end])
    return recommendation, clause


_CANNOT_USE = re.compile(
    r"\bcannot\s+(?:use|take|receive|give|start)\s+([A-Za-z][A-Za-z0-9-]{3,})",
    re.I,
)
_PREFER_OVER = re.compile(
    r"\bprefer\s+[A-Za-z][A-Za-z0-9-]{3,}\s+over\s+([A-Za-z][A-Za-z0-9-]{3,})",
    re.I,
)
_INSTEAD_OF = re.compile(
    r"\b(?:instead of|rather than|preferred to)\s+([A-Za-z][A-Za-z0-9-]{3,})",
    re.I,
)
_SWITCH_FROM = re.compile(
    r"\bswitch\s+from\s+([A-Za-z][A-Za-z0-9-]{3,})",
    re.I,
)
_CHANGE_FROM = re.compile(
    r"\bchange\s+from\s+([A-Za-z][A-Za-z0-9-]{3,})",
    re.I,
)
_REPLACE_WITH = re.compile(
    r"\breplace\s+([A-Za-z][A-Za-z0-9-]{3,})\s+with\b",
    re.I,
)
_STOP_DRUG = re.compile(
    r"(?<!not )\bstop\s+([A-Za-z][A-Za-z0-9-]{3,})",
    re.I,
)


def _drop_alternative_drug(text: str) -> str:
    """Drop a drug the recommendation says not to use when another drug remains."""
    words = [match.group(1) for match in _CANNOT_USE.finditer(text or "")]
    for pattern in (_PREFER_OVER, _INSTEAD_OF, _SWITCH_FROM, _CHANGE_FROM, _REPLACE_WITH, _STOP_DRUG):
        words.extend(match.group(1) for match in pattern.finditer(text or ""))
    kept = text or ""
    for word in words:
        if not intervention_tokens(word):
            continue
        trial = re.sub(rf"\b{re.escape(word)}\b", "", kept, count=1, flags=re.I)
        if intervention_tokens(trial):
            kept = trial
    return kept


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
            recommendation, clause = _caution_clause(sentence)
            recommendation_drugs = intervention_tokens(recommendation)
            clause_drugs = intervention_tokens(clause)
            if recommendation_drugs and clause_drugs and recommendation_drugs.isdisjoint(clause_drugs):
                other.append(recommendation)
                caution.append(clause)
            else:
                caution.append(sentence.strip())
        else:
            other.append(sentence.strip())
    other_text = " ".join(other)
    caution_text = " ".join(caution)
    other_drugs = intervention_tokens(other_text)
    caution_drugs = intervention_tokens(caution_text)
    if other_drugs and caution_drugs and other_drugs.isdisjoint(caution_drugs):
        return _drop_alternative_drug(other_text)
    return _drop_alternative_drug(text)


PATIENT_CHART_ID = "patient-chart"


_CONTINUE_VERB = re.compile(r"\b(?:continue|continuing|continued)\b", re.I)
_BE_CONTINUED = re.compile(r"\b(?:be|was|is|were|been)\s+continued\b", re.I)
_TO_CONTINUE = re.compile(r"\bto\s+(?:be\s+)?continued?\b", re.I)
_REMAIN_ON = re.compile(
    r"\b(?:remain|remains|stay|stays|still|currently|now|back)\s+on\b",
    re.I,
)
_BARE_ON = re.compile(r"(?:^|\bpatient\s+)on\b", re.I)
_TAKING = re.compile(
    r"(?:^|\b(?:is|was|were|are|am|been|being|patient)\s+)taking\b",
    re.I,
)
_TAKES_AT_HOME = re.compile(r"\btakes\b", re.I)
_AT_HOME = re.compile(r"\bat\s+home\b", re.I)
_HOME_DRUG = re.compile(r"\b(?:at\s+)?home\s+", re.I)
_FACILITY_HOME = re.compile(r"\b(?:nursing|care|residential|aged)\s+home\b", re.I)
_RECEIVING = re.compile(
    r"(?:^|\b(?:is|was|were|are|am|been|being|patient)\s+)receiv(?:e|es|ed|ing)\b",
    re.I,
)
_USING = re.compile(
    r"(?:^|\b(?:is|was|were|are|am|been|being|patients?)\s+)using\b"
    r"|(?:^|\bpatients?\s+)uses\b",
    re.I,
)
_TOLERATING = re.compile(
    r"(?:^|\b(?:is|was|were|are|am|been|being|patients?|poorly|well)\s+)tolerat(?:e|es|ed|ing)\b",
    re.I,
)
_COMPLIANT_WITH = re.compile(
    r"\b(?:compliant|adherent)\b(?:\s+\w+){0,3}\s+(?:with|to)\b",
    re.I,
)
_ANTICOAGULATED_WITH = re.compile(
    r"\banticoagulat(?:ed|ing)\b(?:\s+\w+){0,4}\s+with\b",
    re.I,
)
_HAS_DRUG = re.compile(
    r"(?:^|\b(?:patients?|is|was|were|are|am|been)\s+)ha(?:s|ve)\b",
    re.I,
)
_NOT_HAS_OBJECT = re.compile(
    r"^(?:allerg(?:y|ies|ic)|toxicity|histories|history|reactions?|intolerances?|problems?)$",
    re.I,
)
_COMPLIANT_WORD = re.compile(r"\b(?:compliant|adherent)\b", re.I)
_ARRIVAL_ON = re.compile(
    r"\b(?:came|come|comes|coming|admitted|admission|discharg(?:e|ed|ing)|"
    r"presents?|presented|presenting|brought|went|goes|going|gone|sent|sends|sending)\b"
    r"(?:\s+\w+){0,5}\s+on\b"
    r"|^\s*home\s+on\b",
    re.I,
)
_BROUGHT_IN = re.compile(
    r"\bbrought\b(?:\s+(?:in|his|her|their|the|a|an))?\b",
    re.I,
)
_OWN_MED = re.compile(
    r"\b(?:patient(?:'s|s')?|his|her|their)\s+own\b(?:\s+supply\s+of\b)?"
    r"|^\s*own\s+(?:supply\s+of\b)?",
    re.I,
)
_DEPENDENT_ON = re.compile(r"\bdepend(?:s|ed|ent|ing)?\b(?:\s+\w+){0,3}\s+on\b", re.I)
_ESTABLISHED_ON = re.compile(
    r"\b(?:established|stabilised|stabilized|stable)\b(?:\s+\w+){0,3}\s+on\b",
    re.I,
)
_DURATION_DRUG = re.compile(
    r"(?:^|\b(?:is|was|were|are|am|been)\s+)"
    r"(?:long-?term|regular|chronic|maintenance|current(?:ly)?)\s+",
    re.I,
)
_IS_REGULAR_MED = re.compile(
    r"\b(?:is|was|were|are|am|been)\s+"
    r"(?:(?:a|an|the|his|her|their)\s+)?"
    r"(?:long-?term|regular|current|usual|chronic|maintenance|ongoing)\s+"
        r"(?:medications?|medicines?|meds?)\b",
    re.I,
)
_AS_DIRECTED = re.compile(r"\bas\s+(?:directed|prescribed|charted)\b", re.I)
_DOSE_WORD = re.compile(
    r"^(?:\d+(?:\.\d+)?|mg|mcg|g|ml|units?|daily|oral|nocte|mane|bd|tds|qds|po|the|a|an|to|of|for|please)$",
    re.I,
)
_NOT_DIRECTED = re.compile(
    r"\b(?:study|trial|meeting|slide|protocol|guidelines?|paper|review)\b",
    re.I,
)
_COUNSEL_ON = re.compile(
    r"\b(?:update[sd]?|updating|advis(?:e|es|ed|ing)|inform(?:s|ed|ing)?|"
    r"tell(?:s|ing)?|told|counsel(?:s|led|ling|ing)?|discuss(?:es|ed|ing)?|"
    r"speak|spoke|talk(?:s|ed|ing)?|educat(?:e|es|ed|ing)|ask(?:s|ed|ing)?)\b",
    re.I,
)
_PLANNED_FOR = re.compile(
    r"\b(?:is|was|were|are|am)\s+for\b"
    r"|\b(?:is|was|were|are|am)\s+to\s+have\b"
    r"|\b(?:charted|written\s+up|due)\s+for\b"
    r"|\b(?:needs?|requir(?:e|es|ed))\b",
    re.I,
)
_KEEP_VERB = re.compile(
    r"\b(?:resume[ds]?|resuming|restart(?:ed|s|ing)?|recommence[ds]?|recommencing|"
    r"keep|keeps|kept|keeping|maintain(?:ed|s|ing)?)\b",
    re.I,
)
_ONGOING = re.compile(r"\bongoing\b", re.I)
_MONITOR_NEXT = re.compile(
    r"^(?:to\s+)?(?:monitor(?:ing|ed)?|checks?|checking|observ(?:e|ing|ation)|follow-?up|surveillance|inr|labs?)\b",
    re.I,
)
_MONITOR_NOUN = re.compile(
    r"^(?:monitoring|checks|checking|observation|follow-?up|surveillance|inr|labs?)$",
    re.I,
)
_MONITOR_WORD = re.compile(
    r"\b(?:monitor(?:ing|ed)?|checks?|checking|observ(?:e|ing|ation)|follow-?up|surveillance|inr|labs?)\b",
    re.I,
)
_LEADING_FUNCTION = re.compile(r"^(?:to|the|a|an|his|her|their|this|that|daily|oral)$", re.I)


def _content_words(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9'-]+", text or "")
    content: list[str] = []
    skipping = True
    for word in words:
        if skipping and _LEADING_FUNCTION.match(word):
            continue
        skipping = False
        content.append(word)
        if len(content) >= 4:
            break
    return content


def _dose_before_verb(text: str, match_start: int) -> bool:
    """True when the drug or its dose sits before a passive dose verb."""
    raw = text or ""
    words = raw.split()
    if not words:
        return False
    spans = []
    cursor = 0
    for word in words:
        at = raw.find(word, cursor)
        spans.append(at)
        cursor = at + len(word)
    anchor = 0
    for index, begin in enumerate(spans):
        if begin <= match_start < begin + len(words[index]):
            anchor = index
            break
    before = words[max(0, anchor - 8):anchor]
    if not before:
        return False
    for index, word in enumerate(before):
        if word.lower() != "dose" or index + 2 >= len(before):
            continue
        if before[index + 1].lower() == "of" and intervention_tokens(before[index + 2]):
            return True
    for index, word in enumerate(before):
        if not intervention_tokens(word):
            continue
        nxt = before[index + 1] if index + 1 < len(before) else ""
        if _NOT_DOSE_NOUN.match(nxt):
            continue
        return True
    return False


def _dose_order_object(text: str) -> bool:
    """True when the dose change names the drug, not a level or a clearance."""
    if not _direct_drug_object(text):
        return False
    content = _content_words(text)
    drug_at = 2 if content and content[0].lower() == "dose" else 0
    return not (drug_at + 1 < len(content) and _NOT_DOSE_NOUN.match(content[drug_at + 1]))


def _direct_drug_object(text: str) -> bool:
    """True when the next phrase names the drug, not a note or a lab."""
    content = _content_words(text)
    if not content or _MONITOR_NEXT.match(" ".join(content)):
        return False
    if intervention_tokens(content[0]):
        if len(content) >= 2 and _MONITOR_NOUN.match(content[1]):
            return False
        return True
    return (
        content[0].lower() == "dose"
        and len(content) >= 3
        and content[1].lower() == "of"
        and bool(intervention_tokens(content[2]))
    )


def _object_is_drug(text: str) -> bool:
    """True when the verb object is a drug, not a monitoring phrase."""
    content = _content_words(text)
    if not content or _MONITOR_NEXT.match(" ".join(content)):
        return False
    if len(content) >= 2 and intervention_tokens(content[0]) and _MONITOR_NOUN.match(content[1]):
        return False
    return bool(intervention_tokens(" ".join(content[:3])))


def _subject_is_drug(text: str) -> bool:
    """True when a continued action has a drug as its subject."""
    if _MONITOR_WORD.search(text or ""):
        return False
    return bool(intervention_tokens(text))


def _is_continuation_sentence(sentence: str) -> bool:
    """True when the sentence says to keep a drug."""
    from app.scoring.coverage import (
        _BENIGN_DO_NOT,
        _CAUTION,
        _LOCAL_NEGATION,
        _STATUS_ON,
        _current_med_list,
        _negated_off_keep,
        _negated_source_keep,
        _on_object_is_drug,
        _ungoverned_caution,
    )

    if not intervention_tokens(sentence):
        return False
    if _ungoverned_caution(sentence):
        return False
    if _negated_source_keep(sentence) or _negated_off_keep(sentence):
        return True
    if _current_med_list(sentence):
        return True
    if _bare_patient_on(sentence):
        return True
    if _taking_drug(sentence):
        return True
    if _takes_at_home(sentence):
        return True
    if _home_drug(sentence):
        return True
    if _receiving_drug(sentence):
        return True
    if _using_drug(sentence):
        return True
    if _tolerating_drug(sentence):
        return True
    if _compliant_drug(sentence):
        return True
    if _anticoagulated_with_drug(sentence):
        return True
    if _has_current_drug(sentence):
        return True
    if _arrival_on_drug(sentence):
        return True
    if _brought_in_drug(sentence):
        return True
    if _own_supply(sentence):
        return True
    if _dependent_on_drug(sentence):
        return True
    if _established_on_drug(sentence):
        return True
    if _duration_drug(sentence):
        return True
    if _regular_medication(sentence):
        return True
    if _directed_drug(sentence):
        return True
    for match in _CAUTION.finditer(sentence or ""):
        phrase = match.group(0).lower()
        if phrase in {"do not", "should not"} and _BENIGN_DO_NOT.match(sentence[match.end():]):
            return True
    for match in list(_REMAIN_ON.finditer(sentence or "")) + list(_STATUS_ON.finditer(sentence or "")):
        window = sentence[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window) or re.search(r"\bnever\b", window, re.I):
            continue
        if _on_object_is_drug(sentence[match.end():]):
            return True
    continued = _BE_CONTINUED.search(sentence or "")
    if continued and _subject_is_drug(sentence[: continued.start()]):
        return True
    for match in _CONTINUE_VERB.finditer(sentence or ""):
        if match.group(0).lower() == "continued" and continued:
            continue
        if _object_is_drug(sentence[match.end():]):
            return True
    for match in _KEEP_VERB.finditer(sentence or ""):
        window = sentence[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        tail = sentence[match.end():]
        prefix = sentence[: match.start()]
        prefix_drugs = intervention_tokens(prefix)
        tail_drugs = intervention_tokens(tail)
        if _object_is_drug(tail) and (not prefix_drugs or not prefix_drugs.isdisjoint(tail_drugs)):
            return True
        if (
            prefix_drugs
            and not tail_drugs
            and _subject_is_drug(prefix)
            and not re.search(r"\b(?:hold|held|stop(?:ped)?)\b", tail or "", re.I)
        ):
            return True
    to_continue = _TO_CONTINUE.search(sentence or "")
    if to_continue and _subject_is_drug(sentence[: to_continue.start()]):
        return True
    for match in _ONGOING.finditer(sentence or ""):
        if _direct_drug_object(sentence[match.end():]) or _subject_is_drug(sentence[: match.start()]):
            return True
    return False


_RECOMMEND_VERB = re.compile(
    r"\b(?:recommend(?:ed|s|ing)?|first[- ]line|indicat(?:ed|ion)|initiat(?:e|ed|ing)|prescrib(?:e|ed|ing)|start(?:ed|ing)?|prefer(?:red|s)?)\b",
    re.I,
)
_ORDER_VERB = re.compile(
    r"(?:^|[.!?]\s+)(?:please\s+)?(?:use|give)\s+(?:the\s+|a\s+)?[A-Za-z]",
    re.I,
)
_START_VERB = re.compile(
    r"\b(?:commence|commenced|commencing|begin|begins|began|begun|beginning|offer|offered|offering)\b",
    re.I,
)
_ADMIN_VERB = re.compile(r"\b(?:given|gave|used|administer(?:ed|s|ing)?)\b", re.I)
_SUPPLY_VERB = re.compile(
    r"\b(?:suppl(?:y|ies|ied|ying)|dispens(?:e|es|ed|ing))\b",
    re.I,
)
_CHART_ORDER_VERB = re.compile(
    r"\b(?:order(?:s|ed|ing)?|chart(?:s|ed|ing)?|issu(?:e|es|ed|ing)|provid(?:e|es|ed|ing))\b",
    re.I,
)
_RX_MARK = re.compile(r"\brx\b\s*:?", re.I)
_SCRIPT_FOR = re.compile(
    r"\bscript(?:ed|s|ing)?\b(?:\s+\w+){0,3}\s+for\b|\bscripted\b",
    re.I,
)
_SCRIPT_END = re.compile(r"\bscript\s*[.!?]*\s*$", re.I)
_PUT_ON = re.compile(
    r"\b(?:put|puts|putting|placed?|placing)\b(?:\s+\w+){0,5}\s+on\b",
    re.I,
)


def _bare_patient_on(sentence: str) -> bool:
    """True when a chart line says the patient is on a drug without a copula."""
    from app.scoring.coverage import _LOCAL_NEGATION, _on_object_is_drug

    text = sentence or ""
    if _COUNSEL_ON.search(text) or _PUT_ON.search(text):
        return False
    for match in _BARE_ON.finditer(text):
        window = text[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window) or re.search(r"\bnever\b", window, re.I):
            continue
        tail = text[match.end():]
        if re.match(r"\s*(?:hold|the\s+hold)\b", tail, re.I):
            continue
        if _on_object_is_drug(tail):
            return True
    return False


def _taking_drug(sentence: str) -> bool:
    """True when the patient is taking a named drug."""
    from app.scoring.coverage import _LOCAL_NEGATION

    text = sentence or ""
    if _COUNSEL_ON.search(text):
        return False
    for match in _TAKING.finditer(text):
        window = text[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window) or re.search(r"\bnever\b", window, re.I):
            continue
        if _object_is_drug(text[match.end():]):
            return True
    return False


def _takes_at_home(sentence: str) -> bool:
    """True when the patient takes a named drug at home."""
    from app.scoring.coverage import _LOCAL_NEGATION, _ON_TOPIC, _on_object_is_drug

    text = sentence or ""
    if _COUNSEL_ON.search(text):
        return False
    for match in _TAKES_AT_HOME.finditer(text):
        window = text[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window) or re.search(r"\bnever\b", window, re.I):
            continue
        home = _AT_HOME.search(text, match.end())
        if home is None:
            continue
        between = text[match.end(): home.start()]
        if not _on_object_is_drug(between):
            continue
        words = re.findall(r"[A-Za-z0-9'-]+", between)
        content = [word for word in words if word.lower() not in {"to", "the", "a", "an", "his", "her", "their", "this", "that"}]
        nxt = content[1] if len(content) >= 2 else ""
        if nxt and _ON_TOPIC.match(nxt):
            continue
        return True
    return False


def _home_subject_is_drug(prefix: str) -> bool:
    """True when the drug is the subject before at home."""
    words = re.findall(r"[A-Za-z0-9'-]+", prefix or "")
    if words and words[-1].lower() in {"is", "was", "were", "are", "am", "been"}:
        words = words[:-1]
    if not words or not intervention_tokens(words[-1]):
        return False
    if intervention_tokens(words[0]):
        return True
    leaders = {"the", "a", "an", "his", "her", "their"}
    return len(words) >= 2 and words[0].lower() in leaders and bool(intervention_tokens(words[1]))


def _home_drug(sentence: str) -> bool:
    """True when a home line names the current drug."""
    from app.scoring.coverage import _LOCAL_NEGATION, _MED_HISTORY, _MED_TOPIC, _ON_TOPIC, _on_object_is_drug

    text = sentence or ""
    if _COUNSEL_ON.search(text) or _MED_TOPIC.search(text) or _MED_HISTORY.search(text):
        return False
    for match in _AT_HOME.finditer(text):
        window = text[max(0, match.start() - 48): match.start()]
        prefix = text[: match.start()]
        tail = text[match.end():]
        if _LOCAL_NEGATION.search(window) or re.search(r"\bnever\b", window, re.I):
            continue
        tail_words = re.findall(r"[A-Za-z0-9'-]+", tail)
        if any(_ON_TOPIC.match(word) for word in tail_words):
            continue
        if intervention_tokens(tail) and intervention_tokens(prefix).isdisjoint(intervention_tokens(tail)):
            continue
        if _home_subject_is_drug(prefix):
            return True
    for match in _HOME_DRUG.finditer(text):
        if _FACILITY_HOME.search(text[max(0, match.start() - 16): match.end()]):
            continue
        window = text[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window) or re.search(r"\bnever\b", window, re.I):
            continue
        if _on_object_is_drug(text[match.end():]):
            return True
    return False


def _receiving_drug(sentence: str) -> bool:
    """True when the patient is receiving a named drug."""
    from app.scoring.coverage import _LOCAL_NEGATION, _ON_TOPIC

    text = sentence or ""
    if _COUNSEL_ON.search(text):
        return False
    if not re.search(r"\bpatients?\b", text, re.I) and not re.match(r"\s*receiv", text, re.I):
        return False
    for match in _RECEIVING.finditer(text):
        window = text[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window) or re.search(r"\bnever\b", window, re.I):
            continue
        tail = text[match.end():]
        if not _direct_drug_object(tail):
            continue
        content = _content_words(tail)
        nxt = content[1] if len(content) >= 2 else ""
        if nxt and _ON_TOPIC.match(nxt):
            continue
        return True
    return False


def _using_drug(sentence: str) -> bool:
    """True when the patient is using a named drug."""
    from app.scoring.coverage import _LOCAL_NEGATION, _ON_TOPIC

    text = sentence or ""
    if _COUNSEL_ON.search(text):
        return False
    if not re.search(r"\bpatients?\b", text, re.I) and not re.match(r"\s*us(?:ing|es)\b", text, re.I):
        return False
    for match in _USING.finditer(text):
        window = text[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window) or re.search(r"\bnever\b", window, re.I):
            continue
        tail = text[match.end():]
        if not _direct_drug_object(tail):
            continue
        content = _content_words(tail)
        nxt = content[1] if len(content) >= 2 else ""
        if nxt and _ON_TOPIC.match(nxt):
            continue
        return True
    return False


def _tolerating_drug(sentence: str) -> bool:
    """True when the patient is tolerating a named drug."""
    from app.scoring.coverage import _LOCAL_NEGATION, _ON_TOPIC

    text = sentence or ""
    if _COUNSEL_ON.search(text):
        return False
    started = bool(re.match(r"\s*(?:poorly\s+|well\s+)?tolerat", text, re.I))
    for match in _TOLERATING.finditer(text):
        window = text[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window) or re.search(r"\bnever\b", window, re.I):
            continue
        tail = text[match.end():]
        prefix = text[: match.start()]
        if _direct_drug_object(tail):
            if not re.search(r"\bpatients?\b", text, re.I) and not started:
                continue
            content = _content_words(tail)
            nxt = content[1] if len(content) >= 2 else ""
            if nxt and _ON_TOPIC.match(nxt):
                continue
            return True
        if (
            intervention_tokens(prefix)
            and not intervention_tokens(tail)
            and _subject_is_drug(prefix)
            and not _MONITOR_WORD.search(tail or "")
        ):
            return True
    return False


def _compliant_drug(sentence: str) -> bool:
    """True when the patient is compliant or adherent with a named drug."""
    from app.scoring.coverage import _LOCAL_NEGATION, _on_object_is_drug

    text = sentence or ""
    if _COUNSEL_ON.search(text):
        return False
    started = bool(re.match(r"\s*(?:compliant|adherent)\b", text, re.I))
    for match in _COMPLIANT_WITH.finditer(text):
        window = text[max(0, match.start() - 48): match.start()]
        prefix = text[: match.start()]
        if _LOCAL_NEGATION.search(window) or re.search(r"\bnever\b", window, re.I):
            continue
        if re.search(r"\bnon-?$", prefix, re.I):
            continue
        if not re.search(r"\bpatients?\b", text, re.I) and not started:
            continue
        if _on_object_is_drug(text[match.end():]):
            return True
    for match in _COMPLIANT_WORD.finditer(text):
        prefix = text[: match.start()]
        tail = text[match.end():]
        window = text[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window) or re.search(r"\bnever\b", window, re.I):
            continue
        if re.search(r"\bnon-?$", prefix, re.I):
            continue
        if intervention_tokens(tail) or _MONITOR_WORD.search(tail or ""):
            continue
        if intervention_tokens(prefix) and _subject_is_drug(prefix):
            return True
    return False


def _anticoagulated_with_drug(sentence: str) -> bool:
    """True when the patient is anticoagulated with a named drug."""
    from app.scoring.coverage import _LOCAL_NEGATION, _on_object_is_drug

    text = sentence or ""
    if _COUNSEL_ON.search(text):
        return False
    started = bool(re.match(r"\s*anticoagulat(?:ed|ing)\b", text, re.I))
    if not re.search(r"\bpatients?\b", text, re.I) and not started:
        return False
    for match in _ANTICOAGULATED_WITH.finditer(text):
        window = text[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window) or re.search(r"\bnever\b", window, re.I):
            continue
        if _on_object_is_drug(text[match.end():]):
            return True
    return False


def _has_current_drug(sentence: str) -> bool:
    """True when the patient has a named drug, not an allergy or a history."""
    from app.scoring.coverage import _LOCAL_NEGATION, _ON_TOPIC

    text = sentence or ""
    if _COUNSEL_ON.search(text):
        return False
    started = bool(re.match(r"\s*ha(?:s|ve)\b", text, re.I))
    if not re.search(r"\bpatients?\b", text, re.I) and not started:
        return False
    for match in _HAS_DRUG.finditer(text):
        window = text[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window) or re.search(r"\bnever\b", window, re.I):
            continue
        tail = text[match.end():]
        if re.match(r"\s*(?:no|not)\b", tail, re.I):
            continue
        if not _direct_drug_object(tail):
            continue
        content = _content_words(tail)
        nxt = content[1] if len(content) >= 2 else ""
        if nxt and (_ON_TOPIC.match(nxt) or _NOT_HAS_OBJECT.match(nxt)):
            continue
        return True
    return False


def _arrival_on_drug(sentence: str) -> bool:
    """True when the patient arrived or left on a named drug."""
    from app.scoring.coverage import _LOCAL_NEGATION, _on_object_is_drug

    text = sentence or ""
    if _COUNSEL_ON.search(text):
        return False
    for match in _ARRIVAL_ON.finditer(text):
        window = text[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window) or re.search(r"\bnever\b", window, re.I):
            continue
        if _on_object_is_drug(text[match.end():]):
            return True
    return False


def _brought_in_drug(sentence: str) -> bool:
    """True when the patient brought in a named drug."""
    from app.scoring.coverage import _LOCAL_NEGATION, _MED_HISTORY, _ON_TOPIC, _on_object_is_drug

    text = sentence or ""
    if _COUNSEL_ON.search(text) or _MED_HISTORY.search(text):
        return False
    if not re.search(r"\bpatients?\b", text, re.I) and not re.match(r"\s*brought\b", text, re.I):
        return False
    for match in _BROUGHT_IN.finditer(text):
        window = text[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window) or re.search(r"\bnever\b", window, re.I):
            continue
        tail = text[match.end():]
        if not _on_object_is_drug(tail):
            continue
        words = re.findall(r"[A-Za-z0-9'-]+", tail)
        content = [
            word
            for word in words
            if word.lower() not in {"to", "the", "a", "an", "his", "her", "their", "this", "that", "in"}
        ]
        nxt = content[1] if len(content) >= 2 else ""
        if nxt and (_ON_TOPIC.match(nxt) or _NOT_HAS_OBJECT.match(nxt)):
            continue
        return True
    return False


def _own_supply(sentence: str) -> bool:
    """True when the patient's own supply names a drug."""
    from app.scoring.coverage import _LOCAL_NEGATION, _MED_HISTORY, _ON_TOPIC, _on_object_is_drug

    text = sentence or ""
    if _COUNSEL_ON.search(text) or _MED_HISTORY.search(text):
        return False
    for match in _OWN_MED.finditer(text):
        window = text[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window) or re.search(r"\bnever\b", window, re.I):
            continue
        tail = text[match.end():]
        if not _on_object_is_drug(tail):
            continue
        words = re.findall(r"[A-Za-z0-9'-]+", tail)
        content = [
            word
            for word in words
            if word.lower() not in {"to", "the", "a", "an", "his", "her", "their", "this", "that", "of"}
        ]
        nxt = content[1] if len(content) >= 2 else ""
        if nxt and (_ON_TOPIC.match(nxt) or _NOT_HAS_OBJECT.match(nxt)):
            continue
        return True
    return False


def _dependent_on_drug(sentence: str) -> bool:
    """True when the patient depends on a named drug."""
    from app.scoring.coverage import _LOCAL_NEGATION, _on_object_is_drug

    text = sentence or ""
    if _COUNSEL_ON.search(text):
        return False
    if not re.search(r"\bpatient\b", text, re.I) and not re.match(r"\s*depend", text, re.I):
        return False
    for match in _DEPENDENT_ON.finditer(text):
        window = text[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window) or re.search(r"\bnever\b", window, re.I):
            continue
        if _on_object_is_drug(text[match.end():]):
            return True
    return False


def _established_on_drug(sentence: str) -> bool:
    """True when the patient is established or stable on a named drug."""
    from app.scoring.coverage import _LOCAL_NEGATION, _on_object_is_drug

    text = sentence or ""
    if _COUNSEL_ON.search(text):
        return False
    if not re.search(r"\bpatients?\b", text, re.I) and not re.match(
        r"\s*(?:well\s+)?(?:established|stabilis(?:e|ed)|stabiliz(?:e|ed)|stable)\b",
        text,
        re.I,
    ):
        return False
    for match in _ESTABLISHED_ON.finditer(text):
        window = text[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window) or re.search(r"\bnever\b", window, re.I):
            continue
        if _on_object_is_drug(text[match.end():]):
            return True
    return False


def _duration_drug(sentence: str) -> bool:
    """True when a duration word names the current drug."""
    from app.scoring.coverage import _LOCAL_NEGATION, _MED_HISTORY, _on_object_is_drug

    text = sentence or ""
    if _COUNSEL_ON.search(text):
        return False
    for match in _DURATION_DRUG.finditer(text):
        window = text[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window) or re.search(r"\bnever\b", window, re.I):
            continue
        if re.search(r"\bcurrent", match.group(0), re.I) and _MED_HISTORY.search(text[: match.start()]):
            continue
        if _on_object_is_drug(text[match.end():]):
            return True
    return False


def _subject_word_is_drug(prefix: str) -> bool:
    """True when the subject just before the verb is a drug."""
    words = re.findall(r"[A-Za-z0-9'-]+", prefix or "")
    if not words or not intervention_tokens(words[-1]):
        return False
    if intervention_tokens(words[0]):
        return True
    leaders = {"the", "a", "an", "his", "her", "their"}
    return len(words) >= 2 and words[0].lower() in leaders and bool(intervention_tokens(words[1]))


def _regular_medication(sentence: str) -> bool:
    """True when the drug itself is named as a regular medication."""
    from app.scoring.coverage import _LOCAL_NEGATION, _MED_HISTORY, _MED_TOPIC

    text = sentence or ""
    if _COUNSEL_ON.search(text) or _MED_TOPIC.search(text):
        return False
    for match in _IS_REGULAR_MED.finditer(text):
        window = text[max(0, match.start() - 48): match.start()]
        prefix = text[: match.start()]
        tail = text[match.end():]
        if _LOCAL_NEGATION.search(window) or re.search(r"\bnever\b", window, re.I):
            continue
        if _MED_HISTORY.search(prefix):
            continue
        if intervention_tokens(tail) and intervention_tokens(prefix).isdisjoint(intervention_tokens(tail)):
            continue
        if _subject_word_is_drug(prefix):
            return True
    return False


def _directed_drug(sentence: str) -> bool:
    """True when the chart says to give a named drug as directed."""
    text = sentence or ""
    if _COUNSEL_ON.search(text) or _MONITOR_WORD.search(text) or _NOT_DIRECTED.search(text):
        return False
    match = _AS_DIRECTED.search(text)
    if match is None:
        return False
    words = re.findall(r"[A-Za-z0-9'-]+", text[: match.start()])
    kept = [word for word in words if not _DOSE_WORD.match(word)]
    if not kept:
        return False
    order = {"take", "takes", "taking", "give", "gives", "given", "use", "uses", "used"}
    drugs = [word for word in kept if intervention_tokens(word)]
    others = [word for word in kept if word not in drugs]
    if not drugs:
        return False
    return all(word.lower() in order for word in others)


_WITH_DRUG = re.compile(
    r"\b(?:loads?|loaded|loading)\b(?:\s+\w+){0,5}\s+with\b"
    r"|\b(?:cover(?:s|ed|ing)?|bridg(?:e|es|ed|ing))\s+the\s+patient\s+with\b"
    r"|\b(?:was|is|were|been)\s+(?:covered|bridged)\s+with\b",
    re.I,
)
_SWITCH_TO = re.compile(
    r"\b(?:switch|change)\b(?:\s+\w+){0,6}\s+to\b|\breplace\b(?:\s+\w+){0,4}\s+with\b",
    re.I,
)
_DRUG_OF_CHOICE = re.compile(r"\b(?:drug of choice|first choice)\b", re.I)
_DOSE_CHANGE = re.compile(
    r"\b(?:add(?:ed|ing)?|increase[ds]?|increasing|up-?titrates?|up\s+titrates?)\b",
    re.I,
)
_DOSE_REDUCTION = re.compile(
    r"\b(?:decrease[ds]?|decreasing|reduce[ds]?|reducing|lower(?:s|ed|ing)?|"
    r"down-?titrates?|down\s+titrates?|tapers?|tapering)\b",
    re.I,
)
_NOT_DOSE_NOUN = re.compile(
    r"^(?:clearance|levels?|concentrations?|sensitivit(?:y|ies)|effects?|responses?|activity)$",
    re.I,
)
_RECORD_NOT_ORDER = re.compile(
    r"\b(?:allerg(?:y|ies|ic)|intolerance|adverse reaction|problem list)\b",
    re.I,
)


def _rejected_drugs(text: str) -> set[str]:
    words = [match.group(1) for match in _CANNOT_USE.finditer(text or "")]
    for pattern in (_PREFER_OVER, _INSTEAD_OF, _SWITCH_FROM, _CHANGE_FROM, _REPLACE_WITH, _STOP_DRUG):
        words.extend(match.group(1) for match in pattern.finditer(text or ""))
    found: set[str] = set()
    for word in words:
        found.update(intervention_tokens(word))
    return found


def _verb_orders_named_drug(text: str, start: int, end: int) -> bool:
    """True when this verb orders the drug before it or the drug after it."""
    tail = text[end:]
    prefix = text[:start]
    prefix_drugs = intervention_tokens(prefix)
    tail_drugs = intervention_tokens(tail)
    if _object_is_drug(tail) and (not prefix_drugs or not prefix_drugs.isdisjoint(tail_drugs)):
        return True
    return bool(
        prefix_drugs
        and not tail_drugs
        and _subject_is_drug(prefix)
        and not _MONITOR_WORD.search(tail or "")
    )


def _script_orders_drug(text: str) -> bool:
    """True when a script or an Rx names a drug."""
    from app.scoring.coverage import _LOCAL_NEGATION, _ON_TOPIC, _on_object_is_drug

    raw = text or ""
    if _COUNSEL_ON.search(raw):
        return False

    def _blocked(tail: str) -> bool:
        content = _content_words(tail)
        nxt = content[1] if len(content) >= 2 else ""
        return bool(nxt and _ON_TOPIC.match(nxt))

    for match in _RX_MARK.finditer(raw):
        window = raw[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        tail = raw[match.end():]
        drug_tail = re.sub(r"^\s*(?:for\b\s*)?", "", tail, count=1, flags=re.I)
        prefix = raw[: match.start()]
        if _direct_drug_object(drug_tail) and not _blocked(drug_tail):
            return True
        if (
            not intervention_tokens(tail)
            and _subject_is_drug(prefix)
            and not _MONITOR_WORD.search(tail or "")
        ):
            return True
    for match in _SCRIPT_FOR.finditer(raw):
        window = raw[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        tail = raw[match.end():]
        prefix = raw[: match.start()]
        phrase = match.group(0).lower()
        if phrase.endswith(" for"):
            if _on_object_is_drug(tail):
                return True
            continue
        if _direct_drug_object(tail) and not _blocked(tail):
            return True
        if (
            not intervention_tokens(tail)
            and _subject_is_drug(prefix)
            and not _MONITOR_WORD.search(tail or "")
        ):
            return True
    end = _SCRIPT_END.search(raw)
    if end is not None:
        window = raw[max(0, end.start() - 48): end.start()]
        prefix = raw[: end.start()]
        if (
            not _LOCAL_NEGATION.search(window)
            and _subject_is_drug(prefix)
            and not _MONITOR_WORD.search(prefix)
        ):
            return True
    return False


def _has_recommendation_verb(text: str) -> bool:
    from app.scoring.coverage import _LOCAL_NEGATION, _ON_TOPIC, _QUALIFY

    for match in _RECOMMEND_VERB.finditer(text or ""):
        window = text[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        return True
    for match in list(_START_VERB.finditer(text or "")) + list(_ADMIN_VERB.finditer(text or "")):
        window = text[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        if _verb_orders_named_drug(text, match.start(), match.end()):
            return True
    for match in _SUPPLY_VERB.finditer(text or ""):
        window = text[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        tail = text[match.end():]
        prefix = text[: match.start()]
        if _direct_drug_object(tail):
            return True
        if not re.search(r"\b(?:was|is|were|been|be)\s+$", prefix or "", re.I):
            continue
        if (
            intervention_tokens(prefix)
            and not intervention_tokens(tail)
            and _subject_is_drug(prefix)
            and not _MONITOR_WORD.search(tail or "")
        ):
            return True
    for match in _CHART_ORDER_VERB.finditer(text or ""):
        window = text[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        tail = text[match.end():]
        prefix = text[: match.start()]
        word = match.group(0).lower()
        if _direct_drug_object(tail):
            content = _content_words(tail)
            nxt = content[1] if len(content) >= 2 else ""
            if not (nxt and _ON_TOPIC.match(nxt)):
                return True
        if word not in {"ordered", "charted", "issued", "provided"}:
            continue
        if (
            intervention_tokens(prefix)
            and not intervention_tokens(tail)
            and _subject_is_drug(prefix)
            and not _MONITOR_WORD.search(tail or "")
        ):
            return True
    if _script_orders_drug(text):
        return True
    for match in _PUT_ON.finditer(text or ""):
        window = text[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        if _object_is_drug(text[match.end():]):
            return True
    for match in _PLANNED_FOR.finditer(text or ""):
        window = text[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        if _object_is_drug(text[match.end():]):
            return True
    for match in _WITH_DRUG.finditer(text or ""):
        window = text[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        if _MONITOR_WORD.search(text or ""):
            continue
        if _object_is_drug(text[match.end():]):
            return True
    for match in _SWITCH_TO.finditer(text or ""):
        if _object_is_drug(text[match.end():]):
            return True
    for match in list(_DOSE_CHANGE.finditer(text or "")) + list(_DOSE_REDUCTION.finditer(text or "")):
        window = text[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        tail = text[match.end():]
        if _RECORD_NOT_ORDER.search(tail):
            continue
        if _dose_order_object(tail) or _dose_before_verb(text, match.start()):
            return True
    if _DRUG_OF_CHOICE.search(text or "") and intervention_tokens(text) and not _MONITOR_WORD.search(text or ""):
        return True
    if _QUALIFY.search(text or ""):
        return False
    return bool(_ORDER_VERB.search(text or ""))


def _recommendation_sentence(sentence: str) -> str:
    """Keep a chart order for a drug. A dose line and a warning stay out."""
    from app.scoring.coverage import _ungoverned_caution

    if not intervention_tokens(sentence):
        return ""
    recommendation, clause = _caution_clause(sentence)
    recommendation_drugs = intervention_tokens(recommendation)
    clause_drugs = intervention_tokens(clause)
    if recommendation_drugs and clause_drugs and recommendation_drugs.isdisjoint(clause_drugs):
        text = recommendation
    elif _ungoverned_caution(sentence):
        return ""
    else:
        text = sentence
    if intervention_tokens(text) <= _rejected_drugs(text):
        return ""
    kept = _drop_alternative_drug(text)
    if not intervention_tokens(kept):
        return ""
    if not (_has_recommendation_verb(text) or _has_recommendation_verb(kept)):
        return ""
    return " ".join(kept.split())


def _order_verb_spans(sentence: str) -> list[tuple[int, int]]:
    """Return each verb that orders a drug in this sentence."""
    from app.scoring.coverage import _LOCAL_NEGATION

    text = sentence or ""
    found: list[tuple[int, int]] = []
    for pattern in (_DOSE_CHANGE, _DOSE_REDUCTION, _RECOMMEND_VERB, _START_VERB, _SWITCH_TO):
        for match in pattern.finditer(text):
            window = text[max(0, match.start() - 48): match.start()]
            if _LOCAL_NEGATION.search(window):
                continue
            tail = text[match.end():]
            if pattern in {_DOSE_CHANGE, _DOSE_REDUCTION}:
                if _RECORD_NOT_ORDER.search(tail):
                    continue
                if not (_dose_order_object(tail) or _dose_before_verb(text, match.start())):
                    continue
            elif not (
                _object_is_drug(tail)
                or _subject_is_drug(text[: match.start()])
            ):
                continue
            found.append((match.start(), match.end()))
    found.sort()
    kept: list[tuple[int, int]] = []
    for start, end in found:
        if kept and start < kept[-1][1]:
            continue
        kept.append((start, end))
    return kept


_NEXT_AUXILIARY = re.compile(
    r"^(?:was|is|were|been|be|being|should|will|can|may|must)\b",
    re.I,
)


def _split_order_clauses(sentence: str) -> list[str]:
    """Split two orders in one sentence when each order names a different drug."""
    text = " ".join((sentence or "").split())
    verbs = _order_verb_spans(text)
    if len(verbs) < 2:
        return [text] if text else []
    cuts = [0]
    for index in range(1, len(verbs)):
        previous_end = verbs[index - 1][1]
        verb_at = verbs[index][0]
        between = text[previous_end:verb_at]
        cut = verb_at
        cursor = previous_end
        for word in between.split():
            at = text.find(word, cursor)
            if at < 0:
                continue
            cursor = at + len(word)
            if not intervention_tokens(word.strip(".,;:")):
                continue
            rest = text[cursor:verb_at].lstrip(" ,;")
            if rest.lower().startswith("and "):
                rest = rest[4:].lstrip()
            if _NEXT_AUXILIARY.match(rest):
                cut = at
                break
        cuts.append(cut)
    cuts.append(len(text))
    clauses = []
    for start, end in zip(cuts, cuts[1:]):
        piece = text[start:end].strip(" ,;.")
        piece = re.sub(r"^(?:and)\s+", "", piece, count=1, flags=re.I)
        piece = re.sub(r"\s+\band$", "", piece, count=1, flags=re.I).strip()
        if piece and intervention_tokens(piece):
            clauses.append(piece)
    if len(clauses) < 2:
        return [text]
    drugs = [intervention_tokens(clause) for clause in clauses]
    if any(left & right for left, right in zip(drugs, drugs[1:])):
        return [text]
    return clauses


def _shared_order_span(order_text: str, conflict_text: str) -> str:
    """Quote only the order that names the warning drug."""
    shared = intervention_tokens(conflict_text)
    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", order_text or ""):
        for clause in _split_order_clauses(sentence):
            if intervention_tokens(clause) & shared:
                kept.append(clause)
    span = " ".join(kept).strip()
    if span:
        return span
    return " ".join((order_text or "").split())


def _chart_recommendation_text(chart: str) -> str:
    """Keep chart sentences that recommend a drug."""
    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", chart or ""):
        text = _recommendation_sentence(sentence)
        if text:
            kept.append(text)
    return " ".join(kept).strip()


def _kept_continuation(sentence: str) -> str:
    """Keep a continuation. A stop for a different drug stays out of that text."""
    recommendation, clause = _caution_clause(sentence)
    recommendation_drugs = intervention_tokens(recommendation)
    clause_drugs = intervention_tokens(clause)
    if recommendation_drugs and clause_drugs and recommendation_drugs.isdisjoint(clause_drugs):
        text = recommendation
    else:
        text = sentence
    if not _is_continuation_sentence(text):
        return ""
    return " ".join(text.split())


def _chart_continuation_text(chart: str) -> str:
    """Keep chart sentences that say to continue a drug."""
    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", chart or ""):
        text = _kept_continuation(sentence)
        if text:
            kept.append(text)
    return " ".join(kept).strip()


def _chart_caution_text(chart: str) -> str:
    """Keep caution sentences from a chart note. A later drug sentence stays out."""
    from app.retrieval.lexical import LexicalRetriever
    from app.scoring.coverage import (
        _QUALIFY,
        _STOP_ORDER,
        _away_split,
        _come_off_stop,
        _no_longer_stop,
        _off_drug_stop,
        _not_for_stop,
        _not_on_stop,
        _not_receiving_stop,
        _not_using_stop,
        _not_established_stop,
        _not_tolerating_stop,
        _not_compliant_stop,
        _not_anticoagulated_stop,
        _not_have_stop,
        _not_regular_med_stop,
        _not_home_stop,
        _not_brought_stop,
        _not_own_stop,
        _not_current_stop,
        _med_list_stop,
        _refused_stop,
        _negated_away_split,
        _ungoverned_caution,
    )

    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", chart or ""):
        if not sentence.strip():
            continue
        if _ungoverned_caution(sentence) or _QUALIFY.search(sentence):
            if (
                _STOP_ORDER.search(sentence)
                or _away_split(sentence)
                or _negated_away_split(sentence)
                or _come_off_stop(sentence)
                or _no_longer_stop(sentence)
                or _off_drug_stop(sentence)
                or _not_for_stop(sentence)
                or _not_on_stop(sentence)
                or _not_receiving_stop(sentence)
                or _not_using_stop(sentence)
                or _not_established_stop(sentence)
                or _not_tolerating_stop(sentence)
                or _not_compliant_stop(sentence)
                or _not_anticoagulated_stop(sentence)
                or _not_have_stop(sentence)
                or _not_regular_med_stop(sentence)
                or _not_home_stop(sentence)
                or _not_brought_stop(sentence)
                or _not_own_stop(sentence)
                or _not_current_stop(sentence)
                or _med_list_stop(sentence)
                or _refused_stop(sentence)
            ):
                kept.append(named_caution_text(sentence))
            else:
                kept.append(LexicalRetriever._caution_span(sentence))
    return " ".join(kept).strip()


def _unscored_caution(entry: EvidenceLedgerEntry) -> Optional[tuple[str, str]]:
    """Return a caution sentence when the passage has no caution claim."""
    from app.scoring.coverage import _QUALIFY, _ungoverned_caution

    if any(
        claim.polarity
        in {
            EvidencePolarity.QUALIFIES,
            EvidencePolarity.SAFETY,
            EvidencePolarity.CONTRADICTS,
        }
        for claim in entry.claims
    ):
        return None
    text = entry.text or ""
    if _ungoverned_caution(text):
        kind = "warning"
    elif _QUALIFY.search(text):
        kind = "qualify"
    else:
        return None
    conflict = named_caution_text(text)
    if not intervention_tokens(conflict):
        return None
    return kind, conflict


def _unscored_order(entry: EvidenceLedgerEntry) -> Optional[tuple[str, str]]:
    """Return an order sentence when no support claim covers that order."""
    if any(claim.polarity == EvidencePolarity.SUPPORTS for claim in entry.claims):
        return None
    text = entry.text or ""
    recommendation = _chart_recommendation_text(text)
    if recommendation and intervention_tokens(recommendation):
        return "recommendation", recommendation
    continuation = _chart_continuation_text(text)
    if continuation and intervention_tokens(continuation):
        return "continuation", continuation
    return None


def chart_contradictions(chart: str, ledger: List[EvidenceLedgerEntry]) -> List[ContradictionPair]:
    """Pair a chart caution with a selected recommendation for the same drug.

    The chart synopsis is not a retrieved passage. A dose line is not a caution.
    A drug named outside the caution sentence does not form a conflict.
    """
    from app.scoring.coverage import _QUALIFY, _ungoverned_caution

    text = _chart_caution_text(chart)
    continuation = _chart_continuation_text(chart)
    recommendation = _chart_recommendation_text(chart)
    has_caution = bool(text and intervention_tokens(text))
    has_continuation = bool(continuation and intervention_tokens(continuation))
    has_recommendation = bool(recommendation and intervention_tokens(recommendation))
    if not has_caution and not has_continuation and not has_recommendation:
        return []
    caution = _ungoverned_caution(text) if has_caution else False
    qualify = bool(_QUALIFY.search(text)) and not caution if has_caution else False
    pairs: List[ContradictionPair] = []
    seen = set()
    if has_caution and (caution or qualify):
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
            order = _unscored_order(entry)
            if not order:
                continue
            _kind, order_text = order
            if not intervention_tokens(order_text) or not same_intervention(order_text, text):
                continue
            if caution:
                reason = "Same-drug safety warning conflicts with this recommendation."
                severity = "high" if entry.evidence_grade.score >= 0.80 else "medium"
            else:
                reason = "Same-drug caution qualifies this recommendation."
                severity = "low"
            key = ("chart-order", entry.chunk_id, severity)
            if key in seen:
                continue
            seen.add(key)
            span = " ".join(text.split())
            pairs.append(
                ContradictionPair(
                    facet="safety_contraindications",
                    chunk_id_a=entry.chunk_id,
                    chunk_id_b=PATIENT_CHART_ID,
                    severity=severity,
                    reason=(
                        f"Chart caution: {span} | "
                        f"{_reason_with_drug(reason, order_text, text)}"
                    ),
                    grade_a=entry.evidence_grade.score,
                    grade_b=0.0,
                )
            )
    orders = []
    if has_continuation:
        orders.append((continuation, "Chart continuation:"))
    if has_recommendation:
        orders.append((recommendation, "Chart recommendation:"))
    for order_text, label in orders:
        for entry in ledger:
            if entry.chunk_id == PATIENT_CHART_ID:
                continue
            for claim in entry.claims:
                if claim.polarity == EvidencePolarity.QUALIFIES:
                    severity = "low"
                    base = "Same-drug caution qualifies this recommendation."
                elif claim.polarity in {EvidencePolarity.SAFETY, EvidencePolarity.CONTRADICTS}:
                    severity = "high" if entry.evidence_grade.score >= 0.80 else "medium"
                    base = "Same-drug safety warning conflicts with this recommendation."
                else:
                    continue
                conflict_text = _conflict_text(entry, claim)
                if not intervention_tokens(conflict_text) or not same_intervention(order_text, conflict_text):
                    continue
                key = (claim.facet, entry.chunk_id, severity)
                if key in seen:
                    continue
                seen.add(key)
                span = _shared_order_span(order_text, conflict_text)
                pairs.append(
                    ContradictionPair(
                        facet=claim.facet,
                        chunk_id_a=PATIENT_CHART_ID,
                        chunk_id_b=entry.chunk_id,
                        severity=severity,
                        reason=f"{label} {span} | {_reason_with_drug(base, span, conflict_text)}",
                        grade_a=0.0,
                        grade_b=entry.evidence_grade.score,
                    )
                )
            extra = _unscored_caution(entry)
            if not extra:
                continue
            kind, conflict_text = extra
            if not same_intervention(order_text, conflict_text):
                continue
            if kind == "qualify":
                severity = "low"
                base = "Same-drug caution qualifies this recommendation."
            else:
                severity = "high" if entry.evidence_grade.score >= 0.80 else "medium"
                base = "Same-drug safety warning conflicts with this recommendation."
            key = ("passage-text", entry.chunk_id, severity)
            if key in seen:
                continue
            seen.add(key)
            span = _shared_order_span(order_text, conflict_text)
            pairs.append(
                ContradictionPair(
                    facet="safety_contraindications",
                    chunk_id_a=PATIENT_CHART_ID,
                    chunk_id_b=entry.chunk_id,
                    severity=severity,
                    reason=f"{label} {span} | {_reason_with_drug(base, span, conflict_text)}",
                    grade_a=0.0,
                    grade_b=entry.evidence_grade.score,
                )
            )
    return pairs


def _weak_patient_mention(entry: EvidenceLedgerEntry, claim: EvidenceClaim) -> bool:
    """True when a patient word names the drug but the line is not an order."""
    from app.scoring.coverage import _LOCAL_NEGATION

    if claim.facet != "patient_applicability":
        return False
    text = _pairing_text(entry, claim) or (entry.text or "")
    if _has_recommendation_verb(text) or _is_continuation_sentence(text):
        return False
    for match in re.finditer(r"\b(?:takes?|taking)\b", text, re.I):
        if _COUNSEL_ON.search(text):
            continue
        window = text[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        if _object_is_drug(text[match.end():]):
            return False
    if re.search(r"\b(?:lists?|listed|listing)\b", text, re.I) and intervention_tokens(text):
        return False
    return True


def _weak_evidence_mention(entry: EvidenceLedgerEntry, claim: EvidenceClaim) -> bool:
    """True when an evidence word names a drug but the line is not evidence."""
    from app.scoring.coverage import _MED_TOPIC, _current_med_list

    if claim.facet != "evidence_quality":
        return False
    text = _pairing_text(entry, claim) or (entry.text or "")
    if _has_recommendation_verb(text) or _is_continuation_sentence(text) or _current_med_list(text):
        return False
    if not _MED_TOPIC.search(text) or not re.search(r"\b(?:medications?|medicines?|meds)\b", text, re.I):
        return False
    return not re.search(
        r"\b(?:supports?|shows?|showed|found|demonstrat\w*|recommends?|concludes?|concluded)\b",
        text,
        re.I,
    )


def _skip_weak_support(entry: EvidenceLedgerEntry, claim: EvidenceClaim) -> bool:
    """True when a facet keyword names the drug without an order or evidence."""
    return _weak_patient_mention(entry, claim) or _weak_evidence_mention(entry, claim)


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
                        if _skip_weak_support(support, support_claim):
                            continue
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
        if _skip_weak_support(support_entry, support_claim):
            continue
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
        if _skip_weak_support(support_entry, support_claim):
            continue
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
    unscored = []
    for entry in ledger:
        extra = _unscored_caution(entry)
        if extra:
            unscored.append((entry, extra[0], extra[1]))
    for support_entry, support_claim in plain_supports:
        if _skip_weak_support(support_entry, support_claim):
            continue
        support_text = _support_text(support_entry, support_claim)
        if not intervention_tokens(support_text):
            continue
        for caution_entry, kind, conflict_text in unscored:
            if not same_intervention(support_text, conflict_text):
                continue
            if kind == "qualify":
                reason = "Same-drug caution qualifies this recommendation."
                severity = "low"
            else:
                reason = "Same-drug safety warning conflicts with this recommendation."
                severity = None
            _add(
                support_claim.facet,
                support_entry,
                caution_entry,
                _reason_with_drug(reason, support_text, conflict_text),
                severity=severity,
            )
    orders = []
    for entry in ledger:
        extra = _unscored_order(entry)
        if extra:
            orders.append((entry, extra[1]))
    for order_entry, order_text in orders:
        for warning_entry, warning_claim in safety_warnings:
            conflict_text = _conflict_text(warning_entry, warning_claim)
            if not intervention_tokens(conflict_text) or not same_intervention(order_text, conflict_text):
                continue
            _add(
                warning_claim.facet,
                order_entry,
                warning_entry,
                _reason_with_drug(
                    "Same-drug safety warning conflicts with this recommendation.",
                    order_text,
                    conflict_text,
                ),
            )
        for qualify_entry, qualify_claim in qualifies:
            conflict_text = _conflict_text(qualify_entry, qualify_claim)
            if not intervention_tokens(conflict_text) or not same_intervention(order_text, conflict_text):
                continue
            _add(
                qualify_claim.facet,
                order_entry,
                qualify_entry,
                _reason_with_drug(
                    "Same-drug caution qualifies this recommendation.",
                    order_text,
                    conflict_text,
                ),
                severity="low",
            )
        for caution_entry, kind, conflict_text in unscored:
            if not same_intervention(order_text, conflict_text):
                continue
            if kind == "qualify":
                reason = "Same-drug caution qualifies this recommendation."
                severity = "low"
            else:
                reason = "Same-drug safety warning conflicts with this recommendation."
                severity = None
            _add(
                "safety_contraindications",
                order_entry,
                caution_entry,
                _reason_with_drug(reason, order_text, conflict_text),
                severity=severity,
            )
    return pairs
