import { FormEvent, useState } from "react";
import { getTrace, portalUrls, runChat, type Pipeline } from "./api";

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
  dense_score?: number;
};

type Contradiction = {
  facet: string;
  chunk_id_a: string;
  chunk_id_b: string;
  severity: string;
  reason: string;
  resolved?: boolean;
};

function ResultPanel({
  title,
  response,
  trace,
}: {
  title?: string;
  response: any;
  trace?: any;
}) {
  const decision = response?.policy_decision || response?.sufficiency_decision;
  const action = decision?.action || "pending";
  const facets: FacetRow[] = response?.facet_coverage || response?.facet_matrix?.rows || [];
  const passages: Passage[] = response?.evidence_bundle?.passages || [];
  const contradictions: Contradiction[] =
    response?.contradictions || response?.contradiction_ledger?.pairs || [];
  const citations = response?.citations || [];

  return (
    <div className="result">
      {title ? <h2>{title}</h2> : null}
      <div className={`chip ${action}`}>{String(action).replaceAll("_", " ")}</div>
      <p className="meta" style={{ marginTop: 10 }}>
        {response.pipeline || "medswin"} · {response.retrieval_backend || "n/a"} · trace{" "}
        {response.trace_id} · {citations.length} citations · uncertainty{" "}
        {response.uncertainty_level || "n/a"}
      </p>
      <p className="answer" style={{ marginTop: 16 }}>
        {response.answer}
      </p>

      <div style={{ marginTop: 22 }}>
        <h2>Facet coverage</h2>
        <div className="matrix">
          {facets.length === 0 ? (
            <div className="meta">No facet rows — expected for naive-RAG.</div>
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
          {citations.length === 0 ? (
            <div className="meta">No citations attached.</div>
          ) : (
            citations.map((c: any) => (
              <div className="row" key={c.chunk_id || c.doc_id}>
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
          {passages.length === 0 ? (
            <div className="meta">No passages retrieved.</div>
          ) : (
            passages.map((p) => (
              <article className="card" key={p.chunk_id}>
                <strong>
                  {p.source_type} · {p.chunk_id}
                </strong>
                <span className="meta">
                  doc {p.doc_id} · dense {(p.dense_score ?? 0).toFixed(2)} · p̂{" "}
                  {(p.calibrated_score ?? 0).toFixed(2)}
                </span>
                <p style={{ margin: "8px 0 0", lineHeight: 1.45 }}>
                  {p.text.slice(0, 320)}
                  {p.text.length > 320 ? "…" : ""}
                </p>
              </article>
            ))
          )}
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
          <summary>Retrieval / rerank / tools / full JSON</summary>
          <pre>{JSON.stringify(trace || response, null, 2)}</pre>
        </details>
      </div>
    </div>
  );
}

export function App() {
  const [query, setQuery] = useState("");
  const [patientId, setPatientId] = useState("");
  const [orgId, setOrgId] = useState("demo-org");
  const [userId, setUserId] = useState("clinician-1");
  const [pipeline, setPipeline] = useState<Pipeline>("full");
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<any>(null);
  const [compare, setCompare] = useState<any>(null);
  const [trace, setTrace] = useState<any>(null);
  const portals = portalUrls();

  const buttonLabel =
    pipeline === "naive" ? "Ask naive-RAG" : pipeline === "both" ? "Compare both" : "Ask MedSwin";

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setTrace(null);
    setCompare(null);
    setResponse(null);
    try {
      const payload = {
        query,
        user_id: userId,
        org_id: orgId,
        patient_id: patientId || undefined,
        top_k: pipeline === "full" ? undefined : topK,
      };
      const data = await runChat(pipeline, payload);
      if (pipeline === "both") {
        setCompare(data);
      } else {
        setResponse(data);
        if (data.trace_id) {
          try {
            setTrace(await getTrace(data.trace_id, orgId));
          } catch {
            /* chat succeeded; trace optional */
          }
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
          Evidence-gated clinician decision support. Run the full system, the
          naive-RAG control, or both, then inspect the audit on this page.
        </p>
        <nav className="nav">
          <a href={portals.clinician}>Clinician CDS</a>
          <a href={portals.dashboard}>Ops dashboard</a>
          <a href={portals.docs}>OpenAPI</a>
          <a href={portals.eval}>Eval portal</a>
        </nav>
      </header>

      <div className="workspace">
        <form className="panel" onSubmit={onSubmit}>
          <h2>Clinical query</h2>
          <label>Pipeline</label>
          <div className="pipeline">
            {(
              [
                ["full", "Full MedSwin"],
                ["naive", "Naive RAG"],
                ["both", "Compare both"],
              ] as const
            ).map(([value, label]) => (
              <label key={value} className={pipeline === value ? "on" : ""}>
                <input
                  type="radio"
                  name="pipeline"
                  value={value}
                  checked={pipeline === value}
                  onChange={() => setPipeline(value)}
                />
                {label}
              </label>
            ))}
          </div>
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
          {pipeline !== "full" ? (
            <>
              <label htmlFor="topk">Naive top-K</label>
              <input
                id="topk"
                type="number"
                min={1}
                max={20}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value) || 5)}
              />
            </>
          ) : null}
          <button type="submit" disabled={loading || !query.trim()}>
            {loading ? "Running…" : buttonLabel}
          </button>
          {error ? <div className="error">{error}</div> : null}
        </form>

        <section className="panel">
          <h2>Decision support</h2>
          {compare ? (
            <>
              <div className="compare-grid">
                <ResultPanel title="Naive RAG" response={compare.naive || {}} />
                <ResultPanel title="Full MedSwin" response={compare.medswin || {}} />
              </div>
              <div className="drawer">
                <h2>Diff</h2>
                <pre>{JSON.stringify(compare.diff || {}, null, 2)}</pre>
              </div>
            </>
          ) : response ? (
            <ResultPanel response={response} trace={trace} />
          ) : (
            <p className="meta">
              Submit a clinical question to see the answer, evidence, and
              sufficiency decision. Use Compare both to watch naive-RAG against
              the gated pipeline on the same query.
            </p>
          )}
        </section>
      </div>
    </div>
  );
}
