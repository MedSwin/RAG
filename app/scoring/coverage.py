"""Facet alignment and noisy-OR coverage with bootstrap LCB."""

from __future__ import annotations

import math
import random
import re
from typing import Dict, Iterable, List, Optional, Sequence

from app.core.config import settings
from app.schemas.enums import EvidencePolarity, SourceType
from app.schemas.evidence import CandidatePassage, EvidenceClaim, EvidenceLedgerEntry
from app.schemas.facets import ClinicalFacet, FacetCoverage
from app.scoring.calibrate import clamp
from app.scoring.hierarchy import evidence_grade_from_metadata


def _source_policy_score(source_type: SourceType, source_policy: Optional[str]) -> float:
    if not source_policy:
        return 0.0
    if source_policy == "ANY":
        return 0.20
    if source_policy == source_type.value:
        return 0.45
    if source_policy == "CPG" and source_type in {SourceType.LIT, SourceType.SAFETY}:
        return 0.15
    if source_policy == "SAFETY" and source_type in {SourceType.CPG, SourceType.LIT}:
        return 0.20
    return 0.0


_CAUTION_TOKENS = {"allergy", "adverse", "interaction", "avoid", "toxicity"}


def _token_is_caution(token: str) -> bool:
    if token.startswith("contraindicat"):
        return True
    stems = {token}
    if token.endswith("s") and len(token) > 3:
        stems.add(token[:-1])
    return bool(stems & _CAUTION_TOKENS)


def _safety_score(passage: CandidatePassage) -> float:
    text = passage.text.lower()
    hits = 0
    for match in re.finditer(r"[a-z0-9]+", text):
        if not _token_is_caution(match.group(0)):
            continue
        window = text[max(0, match.start() - 48):match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        hits += 1
    if _ungoverned_caution(passage.text):
        hits += 1
    return clamp(hits * 0.18)


def keyword_in_text(keyword: str, text: str) -> bool:
    """Match a keyword as a whole token. A phrase matches as written."""
    needle = (keyword or "").lower().strip()
    if not needle:
        return False
    lowered = (text or "").lower()
    if " " in needle or "-" in needle:
        return needle in lowered
    tokens = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
    if needle in tokens:
        return True
    if needle.endswith("s") and needle[:-1] in tokens:
        return True
    return f"{needle}s" in tokens


def score_passage_facets(
    passage: CandidatePassage,
    facets: Iterable[ClinicalFacet],
) -> Dict[str, float]:
    """Score passage-to-facet alignment (paper Eq. facet-alignment)."""
    text = passage.text.lower()
    scores: Dict[str, float] = {}
    for facet in facets:
        source_bonus = _source_policy_score(passage.source_type, facet.source_policy)
        matched = [keyword for keyword in facet.keywords if keyword_in_text(keyword, text)]
        keyword_score = min(0.65, len(matched) * 0.18)
        safety_bonus = 0.25 if facet.name == "safety_contraindications" and _safety_score(passage) > 0 else 0.0
        if not matched and safety_bonus == 0.0 and source_bonus <= 0.45:
            # Source-policy affinity alone must not mark an unrelated facet as covered.
            # An exact source match is 0.45. It still needs a keyword or a safety hit.
            text_score = 0.0
        else:
            # One keyword does not collect the source bonus. Two keywords, or a safety hit, do.
            # The section heading stays in the fusion rank. It does not raise facet certainty.
            applied_source = source_bonus if len(matched) >= 2 or safety_bonus > 0.0 else 0.0
            text_score = clamp(0.20 + applied_source + keyword_score + safety_bonus)
        scores[facet.name] = text_score
    passage.facet_scores = scores
    return scores


def _binary_entropy(probability: float) -> float:
    p = clamp(probability, 1e-9, 1.0 - 1e-9)
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _contribution(claim: EvidenceClaim, entry: EvidenceLedgerEntry) -> float:
    """Paper coverage term: a_dj * p_hat * g_EBM. No floors that inflate π_j."""
    a_dj = clamp(claim.confidence)
    p_hat = clamp(entry.calibrated_relevance or claim.calibrated_relevance or 0.0)
    g_ebm = clamp(entry.evidence_grade.score if entry.evidence_grade else claim.evidence_grade.score)
    return clamp(a_dj * p_hat * g_ebm)


def _bootstrap_lcb(contributions: Sequence[float], n_boot: int = 64, seed: int = 42) -> float:
    if not contributions:
        return 0.0
    if len(contributions) < 2:
        # One document has no bootstrap sample. Entropy rejects an uncertain probability.
        return clamp(1.0 - math.prod(1.0 - c for c in contributions))
    rng = random.Random(seed)
    samples = []
    n = len(contributions)
    for _ in range(n_boot):
        draw = [contributions[rng.randrange(n)] for _ in range(n)]
        no_support = 1.0
        for c in draw:
            no_support *= 1.0 - clamp(c)
        samples.append(clamp(1.0 - no_support))
    samples.sort()
    idx = max(0, int(0.05 * (len(samples) - 1)))
    return samples[idx]


def compute_facet_coverage(
    facets: List[ClinicalFacet],
    ledger: List[EvidenceLedgerEntry],
) -> List[FacetCoverage]:
    """Compute noisy-OR coverage with bootstrap LCB."""
    coverage: List[FacetCoverage] = []
    for facet in facets:
        contributions: List[float] = []
        supporting: List[str] = []
        contradicting: List[str] = []
        # One contribution per passage, then one per parent document.
        # Heuristic+agent claims on the same chunk must not inflate π_j.
        best_by_chunk: Dict[str, float] = {}
        doc_of_chunk: Dict[str, str] = {}
        for entry in ledger:
            for claim in entry.claims:
                if claim.facet != facet.name:
                    continue
                if claim.polarity == EvidencePolarity.IRRELEVANT:
                    continue
                # A safety warning covers the safety facet. On every other facet it is a conflict.
                safety_conflict = (
                    claim.polarity == EvidencePolarity.SAFETY
                    and facet.name != "safety_contraindications"
                )
                if claim.polarity in {EvidencePolarity.CONTRADICTS, EvidencePolarity.QUALIFIES} or safety_conflict:
                    contradicting.append(entry.chunk_id)
                    continue
                contrib = _contribution(claim, entry)
                prev = best_by_chunk.get(entry.chunk_id)
                if prev is None or contrib > prev:
                    best_by_chunk[entry.chunk_id] = contrib
                    doc_of_chunk[entry.chunk_id] = entry.doc_id or entry.chunk_id
                if claim.polarity != EvidencePolarity.IRRELEVANT:
                    supporting.append(entry.chunk_id)
        best_by_doc: Dict[str, float] = {}
        for chunk_id, contrib in best_by_chunk.items():
            doc_id = doc_of_chunk.get(chunk_id, chunk_id)
            prev = best_by_doc.get(doc_id)
            if prev is None or contrib > prev:
                best_by_doc[doc_id] = contrib
        contributions = list(best_by_doc.values())
        no_support = 1.0
        for c in contributions:
            no_support *= 1.0 - clamp(c)
        probability = clamp(1.0 - no_support)
        lcb = _bootstrap_lcb(contributions)
        entropy = _binary_entropy(probability)
        if lcb >= facet.threshold and entropy <= settings.SUFF_MAX_ENTROPY:
            status = "satisfied"
        elif supporting:
            status = "uncertain"
        elif contradicting:
            status = "contradicted"
        else:
            status = "missing"
        coverage.append(
            FacetCoverage(
                facet=facet.name,
                required=facet.required,
                threshold=facet.threshold,
                coverage_probability=probability,
                lower_confidence_bound=lcb,
                entropy=entropy,
                status=status,
                supporting_chunk_ids=sorted(set(supporting)),
                contradicting_chunk_ids=sorted(set(contradicting)),
            )
        )
    return coverage


_CAUTION = re.compile(r"\b(?:contraindicat\w*|not recommended|\bavoid\b|do not|should not)\b", re.I)
_LOCAL_NEGATION = re.compile(
    r"\b(?:no|not|without|absent|denies|denied|negative for|free of)\b(?:\W+\w+){0,4}\W*$",
    re.I,
)
_BENIGN_DO_NOT = re.compile(
    r"^\s+(?:(?:be|being)\s+)?"
    r"(?:delay|withhold|withheld|wait|hesitate|stop(?:ped)?|discontinue[d]?|cease[d]?|"
    r"interrupt(?:ed)?|hold|held|suspend(?:ed)?|pause[d]?|halt(?:ed)?|"
    r"tapers?|tapered|tapering|weans?|weaned|weaning|withdraw(?:ing|n|s)?|"
    r"omit(?:ted|ting|s)?)\b",
    re.I,
)
_QUALIFY = re.compile(r"\b(?:unless|except|cautions?|precautions?|monitor)\b", re.I)
_NAMED_SAFETY = re.compile(
    r"\b(?:allerg(?:y|ic|ies)|anaphylaxis|anaphylactic|hypersensitivity|adverse|toxicity|interactions?|interacts?|intoleran(?:t|ce)|reactions?|cross-?reactivit(?:y|ies))\b",
    re.I,
)
_STOP_ORDER = re.compile(
    r"\b(?:stop(?:ped)?|withh(?:old|eld)|hold|held|discontinue[d]?|cease[d]?|suspend(?:ed)?|"
    r"pause[d]?|halt(?:ed)?|interrupt(?:ed)?|withdraw(?:ing|n|s)?|omit(?:ted|ting|s)?)\b",
    re.I,
)
_SAFETY_LIST = re.compile(r"^\s+list\b", re.I)


def _token_names_drug(word: str) -> bool:
    """True when one word is a listed drug. This does not read caution polarity."""
    from app.medswin.ledger import _DRUG_ALIASES, _DRUG_SUFFIXES, _DRUG_WORDS, _NOT_DRUGS, _SHORT_ALIASES, _WORD

    token = (word or "").lower().strip(".,;")
    if token in _SHORT_ALIASES:
        return True
    for raw in _WORD.findall(token):
        for part in raw.split("-"):
            if len(part) < 4 or part in _NOT_DRUGS:
                continue
            if part.endswith("s") and part[:-1] in _DRUG_WORDS:
                part = part[:-1]
            if part in _DRUG_ALIASES or part in _DRUG_WORDS or part.endswith(_DRUG_SUFFIXES):
                return True
            if any(len(drug) >= 6 and part.startswith(drug) for drug in _DRUG_WORDS):
                return True
    return False


def _named_safety_caution(text: str) -> bool:
    """True when a safety word names a drug and a nearby negation does not cancel it."""
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
    for match in _NAMED_SAFETY.finditer(raw):
        window = raw[max(0, match.start() - 48):match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        if _SAFETY_LIST.match(raw[match.end():]):
            continue
        anchor = 0
        for index, begin in enumerate(spans):
            if begin <= match.start() < begin + len(words[index]):
                anchor = index
                break
        near = words[max(0, anchor - 4): min(len(words), anchor + 5)]
        if any(_token_names_drug(word) for word in near):
            return True
    for match in _STOP_ORDER.finditer(raw):
        window = raw[max(0, match.start() - 48):match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        anchor = 0
        for index, begin in enumerate(spans):
            if begin <= match.start() < begin + len(words[index]):
                anchor = index
                break
        near = words[max(0, anchor - 4): min(len(words), anchor + 9)]
        if any(_token_names_drug(word) for word in near):
            return True
    return False


def _away_split(text: str) -> Optional[tuple[str, str]]:
    """Split a switch or a replace into the new order and the stopped drug."""
    raw = text or ""
    switched = re.search(
        r"\b(?P<verb>switch|change)\b(?:\s+\w+){0,4}\s+from\s+"
        r"(?P<source>[A-Za-z][A-Za-z0-9-]{3,})\b(?:\s+\w+){0,6}\s+to\s+"
        r"(?P<dest>[A-Za-z][A-Za-z0-9-]{3,})",
        raw,
        re.I,
    )
    replaced = re.search(
        r"\breplace\s+(?P<source>[A-Za-z][A-Za-z0-9-]{3,})\s+with\s+"
        r"(?P<dest>[A-Za-z][A-Za-z0-9-]{3,})",
        raw,
        re.I,
    )
    match = switched or replaced
    if match is None:
        return None
    window = raw[max(0, match.start() - 48):match.start()]
    if _LOCAL_NEGATION.search(window):
        return None
    source = match.group("source")
    dest = match.group("dest")
    if source.lower() == dest.lower():
        return None
    if not _token_names_drug(source) or not _token_names_drug(dest):
        return None
    clause = raw[match.start():match.end("source")].strip(" .,;")
    if match is switched:
        verb = match.group("verb")
        recommendation = f"{verb[:1].upper()}{verb[1:].lower()} to {dest}"
    else:
        recommendation = f"Replace with {dest}"
    return recommendation, clause


def _negated_away_split(text: str) -> Optional[tuple[str, str]]:
    """Keep the current drug and warn for the new drug when the switch is negated."""
    raw = text or ""
    switched = re.search(
        r"\b(?P<verb>switch|change)\b(?:\s+\w+){0,4}\s+from\s+"
        r"(?P<source>[A-Za-z][A-Za-z0-9-]{3,})\b(?:\s+\w+){0,6}\s+to\s+"
        r"(?P<dest>[A-Za-z][A-Za-z0-9-]{3,})",
        raw,
        re.I,
    )
    replaced = re.search(
        r"\breplace\s+(?P<source>[A-Za-z][A-Za-z0-9-]{3,})\s+with\s+"
        r"(?P<dest>[A-Za-z][A-Za-z0-9-]{3,})",
        raw,
        re.I,
    )
    match = switched or replaced
    if match is None:
        return None
    window = raw[max(0, match.start() - 48):match.start()]
    if not _LOCAL_NEGATION.search(window):
        return None
    source = match.group("source")
    dest = match.group("dest")
    if source.lower() == dest.lower():
        return None
    if not _token_names_drug(source) or not _token_names_drug(dest):
        return None
    continuation = f"Continue {source}"
    if match is switched:
        verb = match.group("verb").lower()
        caution = f"Do not {verb} to {dest}"
    else:
        caution = f"Do not replace with {dest}"
    return continuation, caution


def _negated_source_keep(text: str) -> bool:
    """True when a negated switch names only the current drug."""
    if _negated_away_split(text):
        return False
    raw = text or ""
    match = re.search(
        r"\b(?:switch|change)\b(?:\s+\w+){0,4}\s+from\s+([A-Za-z][A-Za-z0-9-]{3,})\b"
        r"|\breplace\s+([A-Za-z][A-Za-z0-9-]{3,})\b",
        raw,
        re.I,
    )
    if match is None:
        return False
    window = raw[max(0, match.start() - 48):match.start()]
    if not _LOCAL_NEGATION.search(window):
        return False
    word = match.group(1) or match.group(2)
    return _token_names_drug(word)


_COME_OFF = re.compile(
    r"\b(?:tapers?|tapered|tapering|weans?|weaned|weaning)\b(?:\s+\w+){0,4}\s+off\b"
    r"|\b(?:weans?|weaned|weaning)\b"
    r"|\b(?:takes?|taken|taking|comes?|came|coming)\b(?:\s+\w+){0,4}\s+off\b"
    r"|\b(?:tapers?|tapered|tapering)\b(?:\s+\w+){0,4}\s+from\s+[A-Za-z][A-Za-z0-9-]{3,}",
    re.I,
)


_NO_LONGER = re.compile(
    r"\bno longer\b(?:\s+\w+){0,3}\s+(?:on|taking|takes|take|prescrib(?:e|ed|ing))\b",
    re.I,
)
_OFF_DRUG = re.compile(
    r"\b(?:is|was|were|are|am|been)\s+off(?:\s+of)?\b"
    r"|\boff(?:\s+of)?(?=\s+[A-Za-z])",
    re.I,
)
_REFUSED = re.compile(
    r"\b(?:declin(?:e|ed|es|ing)|refus(?:e|ed|es|ing))\b",
    re.I,
)
_NOT_FOR = re.compile(
    r"\bnot\s+for\b|\bnot\s+to\s+have\b|\bnot\s+to\s+be\s+given\b"
    r"|\bnot\s+(?:charted|due)\s+for\b|\bnot\s+written\s+up\s+for\b"
    r"|\bnot\s+(?:need|needs|require|requires|required|take|taking)\b"
    r"|\bnot\s+(?:currently|now|still)\s+taking\b",
    re.I,
)
_NOT_RECEIVING = re.compile(
    r"\bnot\s+(?:(?:currently|now|still)\s+)?receiv(?:e|es|ed|ing)\b",
    re.I,
)
_NOT_USING = re.compile(
    r"\bnot\s+(?:(?:currently|now|still)\s+)?(?:using|uses|use|used)\b",
    re.I,
)
_NOT_ESTABLISHED = re.compile(
    r"\bnot\s+(?:(?:yet|currently|now|still)\s+)?(?:established|stabilised|stabilized|stable)\s+on\b"
    r"|\bnever\s+(?:been\s+)?(?:established|stabilised|stabilized|stable)\s+on\b",
    re.I,
)
_NOT_TOLERATING = re.compile(
    r"\bnot\s+(?:(?:currently|now|still)\s+)?tolerat(?:e|es|ed|ing)\b"
    r"|\bunable\s+to\s+tolerate\b"
    r"|\bnever\s+(?:been\s+)?tolerat(?:e|es|ed|ing)\b",
    re.I,
)
_NOT_COMPLIANT = re.compile(
    r"\bnot\s+(?:(?:currently|now|still)\s+)?(?:compliant|adherent)\b(?:\s+\w+){0,3}\s+(?:with|to)\b"
    r"|\bnon-?compliant\b(?:\s+\w+){0,3}\s+with\b"
    r"|\bnon-?adherent\b(?:\s+\w+){0,3}\s+to\b"
    r"|\bnever\s+(?:been\s+)?(?:compliant|adherent)\b(?:\s+\w+){0,3}\s+(?:with|to)\b"
    r"|\bnon-?(?:compliant|adherent)\s*[.!?]*\s*$",
    re.I,
)
_NOT_ANTICOAGULATED = re.compile(
    r"\bnot\s+(?:(?:currently|now|still)\s+)?anticoagulat(?:ed|ing)\b(?:\s+\w+){0,4}\s+with\b"
    r"|\bnever\s+(?:been\s+)?anticoagulat(?:ed|ing)\b(?:\s+\w+){0,4}\s+with\b",
    re.I,
)
_NOT_ON = re.compile(
    r"\bnot\s+on\b"
    r"|\bnot\s+(?:currently|now|still|back)\s+on\b"
    r"|\b(?:is|was|were|are|am|been)\s+(?:not|never)\s+on\b",
    re.I,
)
_STATUS_ON = re.compile(r"\b(?:is|was|were|are|am|been)\s+on\b", re.I)
_ON_TOPIC = re.compile(
    r"^(?:dosing|pathway|protocol|guidelines?|meeting|study|trial|slide|"
    r"discussion|agenda|review|notes?|chart|lists?|levels?|clearances?|"
    r"concentrations?|results?|values?)$",
    re.I,
)
_ON_MODIFIER = re.compile(
    r"^(?:long-?term|regular|daily|oral|chronic|ongoing|maintenance|current|"
    r"usual|therapeutic|low-?dose|high-?dose)$",
    re.I,
)
_MED_LIST = re.compile(
    r"\b(?:medications?|medicines?|meds)(?:\s+list)?\b(?P<mid>(?:\s+\w+){0,4})\s*(?P<mark>includes?\b|:)",
    re.I,
)
_MED_HISTORY = re.compile(r"\b(?:history|past|previously)\b", re.I)
_MED_TOPIC = re.compile(
    r"\b(?:reviews?|discuss(?:ed|ion|ing)?|counsell?(?:ing|ed)?|meetings?|questions?)\b",
    re.I,
)


def _no_longer_stop(text: str) -> bool:
    """True when the patient is no longer on a named drug."""
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
    for match in _NO_LONGER.finditer(raw):
        window = raw[max(0, match.start() - 48):match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        anchor = 0
        for index, begin in enumerate(spans):
            if begin <= match.start() < begin + len(words[index]):
                anchor = index
                break
        near = words[max(0, anchor - 4): min(len(words), anchor + 9)]
        if any(_token_names_drug(word) for word in near):
            return True
    return False


def _pattern_names_drug(text: str, pattern: re.Pattern) -> bool:
    """True when this pattern sits next to a named drug and a negation does not cancel it."""
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
    for match in pattern.finditer(raw):
        window = raw[max(0, match.start() - 48):match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        anchor = 0
        for index, begin in enumerate(spans):
            if begin <= match.start() < begin + len(words[index]):
                anchor = index
                break
        near = words[max(0, anchor - 4): min(len(words), anchor + 9)]
        if any(_token_names_drug(word) for word in near):
            return True
    return False


def _off_drug_stop(text: str) -> bool:
    """True when the patient is off a named drug."""
    return _pattern_names_drug(text, _OFF_DRUG)


def _not_receiving_stop(text: str) -> bool:
    """True when the patient is not receiving a named drug."""
    raw = text or ""
    if re.search(r"\b(?:counsel|discuss|advise|update|inform|educat\w*)\b", raw, re.I):
        return False
    for match in _NOT_RECEIVING.finditer(raw):
        window = raw[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        words = re.findall(r"[A-Za-z0-9'-]+", raw[match.end():])
        content: list[str] = []
        skipping = True
        for word in words:
            if skipping and word.lower() in {"to", "the", "a", "an", "his", "her", "their", "this", "that"}:
                continue
            skipping = False
            content.append(word)
            if len(content) >= 3:
                break
        if not content:
            continue
        if content[0].lower() == "dose" and len(content) >= 3 and content[1].lower() == "of":
            drug = content[2]
            nxt = ""
        elif _token_names_drug(content[0]):
            drug = content[0]
            nxt = content[1] if len(content) > 1 else ""
        else:
            continue
        if not _token_names_drug(drug) or (nxt and _ON_TOPIC.match(nxt)):
            continue
        return True
    return False


def _not_using_stop(text: str) -> bool:
    """True when the patient is not using a named drug."""
    raw = text or ""
    if re.search(r"\b(?:counsel|discuss|advise|update|inform|educat\w*)\b", raw, re.I):
        return False
    for match in _NOT_USING.finditer(raw):
        window = raw[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        words = re.findall(r"[A-Za-z0-9'-]+", raw[match.end():])
        content: list[str] = []
        skipping = True
        for word in words:
            if skipping and word.lower() in {"to", "the", "a", "an", "his", "her", "their", "this", "that"}:
                continue
            skipping = False
            content.append(word)
            if len(content) >= 3:
                break
        if not content:
            continue
        if content[0].lower() == "dose" and len(content) >= 3 and content[1].lower() == "of":
            drug = content[2]
            nxt = ""
        elif _token_names_drug(content[0]):
            drug = content[0]
            nxt = content[1] if len(content) > 1 else ""
        else:
            continue
        if not _token_names_drug(drug) or (nxt and _ON_TOPIC.match(nxt)):
            continue
        return True
    return False


def _not_established_stop(text: str) -> bool:
    """True when the patient is not established on a named drug."""
    raw = text or ""
    if re.search(r"\b(?:counsel|discuss|advise|update|inform|educat\w*)\b", raw, re.I):
        return False
    for match in _NOT_ESTABLISHED.finditer(raw):
        window = raw[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        if _on_object_is_drug(raw[match.end():]):
            return True
    return False


def _not_tolerating_stop(text: str) -> bool:
    """True when the patient is not tolerating a named drug."""
    raw = text or ""
    if re.search(r"\b(?:counsel|discuss|advise|update|inform|educat\w*)\b", raw, re.I):
        return False
    for match in _NOT_TOLERATING.finditer(raw):
        window = raw[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        words = re.findall(r"[A-Za-z0-9'-]+", raw[match.end():])
        content: list[str] = []
        skipping = True
        for word in words:
            if skipping and word.lower() in {"to", "the", "a", "an", "his", "her", "their", "this", "that"}:
                continue
            skipping = False
            content.append(word)
            if len(content) >= 3:
                break
        if not content:
            continue
        if content[0].lower() == "dose" and len(content) >= 3 and content[1].lower() == "of":
            drug = content[2]
            nxt = ""
        elif _token_names_drug(content[0]):
            drug = content[0]
            nxt = content[1] if len(content) > 1 else ""
        else:
            continue
        if not _token_names_drug(drug) or (nxt and _ON_TOPIC.match(nxt)):
            continue
        return True
    return False


def _not_compliant_stop(text: str) -> bool:
    """True when the patient is not compliant with a named drug."""
    raw = text or ""
    if re.search(r"\b(?:counsel|discuss|advise|update|inform|educat\w*)\b", raw, re.I):
        return False
    for match in _NOT_COMPLIANT.finditer(raw):
        window = raw[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        tail = raw[match.end():]
        if _on_object_is_drug(tail):
            return True
        if tail.strip():
            continue
        words = re.findall(r"[A-Za-z0-9'-]+", raw[: match.start()])
        if any(_token_names_drug(word) for word in words):
            return True
    return False


def _not_anticoagulated_stop(text: str) -> bool:
    """True when the patient is not anticoagulated with a named drug."""
    raw = text or ""
    if re.search(r"\b(?:counsel|discuss|advise|update|inform|educat\w*)\b", raw, re.I):
        return False
    for match in _NOT_ANTICOAGULATED.finditer(raw):
        window = raw[max(0, match.start() - 48): match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        if _on_object_is_drug(raw[match.end():]):
            return True
    return False


def _not_for_stop(text: str) -> bool:
    """True when the line says the patient is not for a named drug."""
    if _negated_off_keep(text):
        return False
    if re.search(r"\bnot\s+(?:(?:currently|now|still)\s+)?taking\b", text or "", re.I) and re.search(
        r"\b(?:counsel|discuss|advise|update|inform|educat\w*)\b",
        text or "",
        re.I,
    ):
        return False
    return _pattern_names_drug(text, _NOT_FOR)


def _refused_stop(text: str) -> bool:
    """True when the patient declined or refused a named drug."""
    return _pattern_names_drug(text, _REFUSED)


def _on_object_is_drug(text: str) -> bool:
    """True when the words after on name a drug, not a topic."""
    words = re.findall(r"[A-Za-z0-9'-]+", text or "")
    content: list[str] = []
    skipping = True
    for word in words:
        if skipping and word.lower() in {"to", "the", "a", "an", "his", "her", "their", "this", "that"}:
            continue
        skipping = False
        content.append(word)
        if len(content) >= 3:
            break
    if not content:
        return False
    if _token_names_drug(content[0]):
        drug_at = 0
    elif (
        len(content) >= 2
        and _ON_MODIFIER.match(content[0])
        and _token_names_drug(content[1])
    ):
        drug_at = 1
    else:
        return False
    nxt = content[drug_at + 1] if drug_at + 1 < len(content) else ""
    return not (nxt and _ON_TOPIC.match(nxt))


def _med_list_kind(text: str) -> str:
    """Return keep, stop, or empty for a medication-list line."""
    raw = text or ""
    if _MED_HISTORY.search(raw) or _MED_TOPIC.search(raw):
        return ""
    for match in _MED_LIST.finditer(raw):
        window = raw[max(0, match.start() - 48): match.start("mark")]
        tail = raw[match.end():]
        negated = bool(_LOCAL_NEGATION.search(window) or re.match(r"\s*(?:no|not)\b", tail, re.I))
        drug_tail = re.sub(r"^\s*(?:no|not)\b", "", tail, count=1, flags=re.I) if negated else tail
        if not _on_object_is_drug(drug_tail):
            continue
        return "stop" if negated else "keep"
    return ""


def _current_med_list(text: str) -> bool:
    """True when a medication list names a current drug."""
    return _med_list_kind(text) == "keep"


def _med_list_stop(text: str) -> bool:
    """True when a medication list says the drug is absent."""
    if _negated_off_keep(text):
        return False
    return _med_list_kind(text) == "stop"


def _not_on_stop(text: str) -> bool:
    """True when the patient is not on a named drug."""
    raw = text or ""
    if _negated_off_keep(raw):
        return False
    for match in _NOT_ON.finditer(raw):
        if _on_object_is_drug(raw[match.end():]):
            return True
    for match in _STATUS_ON.finditer(raw):
        window = raw[max(0, match.start() - 48): match.start()]
        if not (_LOCAL_NEGATION.search(window) or re.search(r"\bnever\b", window, re.I)):
            continue
        if _on_object_is_drug(raw[match.end():]):
            return True
    return False


def _come_off_stop(text: str) -> bool:
    """True when taper or wean takes the patient off a named drug."""
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
    for match in _COME_OFF.finditer(raw):
        window = raw[max(0, match.start() - 48):match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        anchor = 0
        for index, begin in enumerate(spans):
            if begin <= match.start() < begin + len(words[index]):
                anchor = index
                break
        from_drug = re.search(
            r"\bfrom\s+([A-Za-z][A-Za-z0-9-]{3,})",
            match.group(0),
            re.I,
        )
        if from_drug is not None:
            if _token_names_drug(from_drug.group(1)):
                return True
            continue
        near = words[max(0, anchor - 4): min(len(words), anchor + 9)]
        if any(_token_names_drug(word) for word in near):
            return True
    return False


def _negated_off_keep(text: str) -> bool:
    """True when a negation keeps the drug that a come-off line would stop."""
    raw = text or ""
    for match in _COME_OFF.finditer(raw):
        window = raw[max(0, match.start() - 48):match.start()]
        if not _LOCAL_NEGATION.search(window):
            continue
        from_drug = re.search(
            r"\bfrom\s+([A-Za-z][A-Za-z0-9-]{3,})",
            match.group(0),
            re.I,
        )
        if from_drug is not None:
            if _token_names_drug(from_drug.group(1)):
                return True
            continue
        words = raw.split()
        spans = []
        cursor = 0
        for word in words:
            at = raw.find(word, cursor)
            spans.append(at)
            cursor = at + len(word)
        anchor = 0
        for index, begin in enumerate(spans):
            if begin <= match.start() < begin + len(words[index]):
                anchor = index
                break
        near = words[max(0, anchor - 4): min(len(words), anchor + 9)]
        if any(_token_names_drug(word) for word in near):
            return True
    return False


def _source_stop(text: str) -> bool:
    """True when the line stops one named drug. The new drug stays out of this check."""
    raw = text or ""
    pattern = re.compile(
        r"\b(?:switch|change)\b(?:\s+\w+){0,4}\s+from\s+([A-Za-z][A-Za-z0-9-]{3,})\b"
        r"|\breplace\s+([A-Za-z][A-Za-z0-9-]{3,})\b",
        re.I,
    )
    for match in pattern.finditer(raw):
        window = raw[max(0, match.start() - 48):match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        word = match.group(1) or match.group(2)
        if _token_names_drug(word):
            return True
    return False


def _ungoverned_caution(text: str) -> bool:
    """True when caution wording is not cancelled by a nearby negation."""
    for match in _CAUTION.finditer(text or ""):
        window = text[max(0, match.start() - 48):match.start()]
        if _LOCAL_NEGATION.search(window):
            continue
        phrase = match.group(0).lower()
        if phrase in {"do not", "should not"} and _BENIGN_DO_NOT.match(text[match.end():]):
            continue
        if phrase in {"do not", "should not"} and (
            _negated_source_keep(text) or _negated_off_keep(text)
        ):
            continue
        return True
    return (
        _named_safety_caution(text)
        or _source_stop(text)
        or _come_off_stop(text)
        or _no_longer_stop(text)
        or _off_drug_stop(text)
        or _not_for_stop(text)
        or _not_on_stop(text)
        or _not_receiving_stop(text)
        or _not_using_stop(text)
        or _not_established_stop(text)
        or _not_tolerating_stop(text)
        or _not_compliant_stop(text)
        or _not_anticoagulated_stop(text)
        or _med_list_stop(text)
        or _refused_stop(text)
    )


def infer_polarity(passage: CandidatePassage, facet: ClinicalFacet) -> EvidencePolarity:
    text = passage.text or ""
    if _ungoverned_caution(text):
        return EvidencePolarity.CONTRADICTS if facet.name != "safety_contraindications" else EvidencePolarity.SAFETY
    if _QUALIFY.search(text):
        return EvidencePolarity.QUALIFIES
    explicit = (passage.metadata or {}).get("polarity")
    if explicit:
        try:
            return EvidencePolarity(explicit)
        except ValueError:
            pass
    return EvidencePolarity.SUPPORTS
