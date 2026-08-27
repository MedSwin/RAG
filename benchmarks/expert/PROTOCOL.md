# T3 clinician protocol

TREC CDS 2016 has no abstain gold. Do not rate “would you answer this ICU note?” without the packed evidence.

## Stimulus

- Ingest the official **note** as EMR (`patient_id = trec-cds-{n}`)
- Query = official type question only
- `include_patient_context_in_query=false`
- Frozen generator: Foundry GPT
- Persist the immutable pack before rating

## Task A — sufficiency

Rater sees: note + type question + packed snippets. No system name, no answer, no “insufficient evidence” branding.

Label: `answer` or `abstain` given *this* bundle. Gold is per (topic, bundle).

## Task B — claim support

Atomic claims from the clinician-visible answer (cap 8; abstain rationale cap 3). Not the MAC ledger.

Label each claim against **cited packed snippets** only: supported / unsupported / contradictory. Uncited = unsupported.

Judge the 2016 snippets, not 2026 knowledge.

## Raters

- Two independent clinicians + one adjudicator
- IM / hospital medicine / critical care
- Blinded to system and generator
- Pilot on TREC CDS 2014/2015 or mock notes — never hold out a 2016 topic
- Report Cohen κ + 95% CI; also Randolph κ if prevalence is extreme

## What is not in this study

- Med-PaLM axes
- Human T4 (−gate / −MAC). Those conditions change the bundle, so Task A gold is not reusable. T4 is automatic rates only.
