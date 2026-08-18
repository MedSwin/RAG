import { FormEvent, useMemo, useState } from "react";
import { chat, getTrace } from "./api";

type FacetRow = {
  facet: string;
  status: string;
  lower_confidence_bound?: number;
  coverage_probability?: number;
  required?: boolean;
};

type Passage = {
  chunk_id: string;
  doc_id: string;
  source_type: string;
  text: string;
  calibrated_score?: number;
  fusion_score?: number;
};

type Contradiction = {
  facet: string;
  chunk_id_a: string;
  chunk_id_b: string;
  severity: string;
  reason: string;
  resolved?: boolean;
};

export function App() {
  const [query, setQuery] = useState("");
  const [patientId, setPatientId] = useState("");
  const [orgId, setOrgId] = useState("demo-org");
  const [userId, setUserId] = useState("clinician-1");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<any>(null);
  const [trace, setTrace] = useState<any>(null);

  const decision = response?.policy_decision || response?.sufficiency_decision;
  const action = decision?.action || "pending";
  const facets: FacetRow[] = response?.facet_coverage || response?.facet_matrix?.rows || [];
  const passages: Passage[] = response?.evidence_bundle?.passages || [];
  const contradictions: Contradiction[] =
    response?.contradictions || response?.contradiction_ledger?.pairs || [];

  const citationCount = useMemo(
    () => (response?.citations || []).length,
    [response]
  );

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setTrace(null);
    try {
      const data = await chat({
        query,
        user_id: userId,
        org_id: orgId,
        patient_id: patientId || undefined,
      });
      setResponse(data);
      if (data.trace_id) {
        try {
          setTrace(await getTrace(data.trace_id, orgId));
        } catch {
          /* chat succeeded; trace optional */
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="shell">
      <header className="brand">
        <h1>MedSwin</h1>
        <p>
          Evidence-gated clinician decision support — retrieve enough, know when
          evidence is insufficient, and expose a reviewable audit trace.
        </p>
      </header>

      <div className="workspace">
        <form className="panel" onSubmit={onSubmit}>
          <h2>Clinical query</h2>
          <label htmlFor="query">Question</label>
          <textarea
            id="query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Can this patient continue metformin after the latest renal-function result?"
            required
          />
          <label htmlFor="patient">Patient ID</label>
          <input
            id="patient"
            value={patientId}
            onChange={(e) => setPatientId(e.target.value)}
            placeholder="optional EMR scope"
          />
          <label htmlFor="org">Organization</label>
          <input id="org" value={orgId} onChange={(e) => setOrgId(e.target.value)} />
          <label htmlFor="user">User</label>
          <input id="user" value={userId} onChange={(e) => setUserId(e.target.value)} />
          <button type="submit" disabled={loading || !query.trim()}>
            {loading ? "Running MedSwin…" : "Ask MedSwin"}
          </button>
          {error ? <div className="error">{error}</div> : null}
        </form>

        <section className="panel">
          <h2>Decision support</h2>
          {response ? (
            <>
              <div className={`chip ${action}`}>{String(action).replaceAll("_", " ")}</div>
              <p className="meta" style={{ marginTop: 10 }}>
                Trace {response.trace_id} · {citationCount} citations · uncertainty{" "}
                {response.uncertainty_level || "n/a"}
              </p>
              <p className="answer" style={{ marginTop: 16 }}>
                {response.answer}
              </p>

              <div style={{ marginTop: 22 }}>
                <h2>Facet coverage</h2>
                <div className="matrix">
                  {facets.length === 0 ? (
                    <div className="meta">No facet rows returned.</div>
                  ) : (
                    facets.map((row) => (
                      <div className="row" key={row.facet}>
                        <strong>
                          {row.facet} · {row.status}
                          {row.required ? " (required)" : ""}
                        </strong>
                        <span className="meta">
                          LCB {(row.lower_confidence_bound ?? 0).toFixed(2)} · π{" "}
                          {(row.coverage_probability ?? 0).toFixed(2)}
                        </span>
                      </div>
                    ))
                  )}
                </div>
              </div>

              <div style={{ marginTop: 22 }}>
                <h2>Citations</h2>
                <div className="evidence">
                  {(response.citations || []).length === 0 ? (
                    <div className="meta">No citations attached.</div>
                  ) : (
                    (response.citations || []).map((c: any) => (
                      <div className="row" key={c.chunk_id}>
                        <strong>
                          {c.source_type} · {c.chunk_id}
                        </strong>
                        <span className="meta">
                          doc {c.doc_id}
                          {c.section ? ` · ${c.section}` : ""}
                          {c.guideline_version ? ` · v${c.guideline_version}` : ""}
                        </span>
                      </div>
                    ))
                  )}
                </div>
              </div>

              <div style={{ marginTop: 22 }}>
                <h2>Evidence bundle</h2>
                <div className="evidence">
                  {passages.map((p) => (
                    <article className="card" key={p.chunk_id}>
                      <strong>
                        {p.source_type} · {p.chunk_id}
                      </strong>
                      <span className="meta">
                        doc {p.doc_id} · p̂ {(p.calibrated_score ?? 0).toFixed(2)} · fusion{" "}
                        {(p.fusion_score ?? 0).toFixed(2)}
                      </span>
                      <p style={{ margin: "8px 0 0", lineHeight: 1.45 }}>
                        {p.text.slice(0, 320)}
                        {p.text.length > 320 ? "…" : ""}
                      </p>
                    </article>
                  ))}
                </div>
              </div>

              <div style={{ marginTop: 22 }}>
                <h2>Contradiction ledger</h2>
                <div className="conflicts">
                  {contradictions.length === 0 ? (
                    <div className="meta">No contradictions recorded.</div>
                  ) : (
                    contradictions.map((c, idx) => (
                      <div className="row" key={`${c.chunk_id_a}-${c.chunk_id_b}-${idx}`}>
                        <strong>
                          {c.facet} · {c.severity}
                          {c.resolved ? " · resolved" : " · unresolved"}
                        </strong>
                        <span className="meta">
                          {c.chunk_id_a} ↔ {c.chunk_id_b}
                        </span>
                        <p style={{ margin: "6px 0 0" }}>{c.reason}</p>
                      </div>
                    ))
                  )}
                </div>
              </div>

              <div className="drawer">
                <h2>Audit trace</h2>
                <details>
                  <summary>Retrieval / rerank / tools / full trace JSON</summary>
                  <pre>{JSON.stringify(trace || response, null, 2)}</pre>
                </details>
              </div>
            </>
          ) : (
            <p className="meta">
              Submit a clinical question to see the answer, facet matrix, evidence,
              contradictions, and sufficiency decision.
            </p>
          )}
        </section>
      </div>
    </div>
  );
}
