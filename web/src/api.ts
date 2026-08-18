const API_BASE = import.meta.env.VITE_API_BASE || "/api/v1";

export type Pipeline = "full" | "naive" | "both";

export type ChatPayload = {
  query: string;
  user_id: string;
  org_id: string;
  session_id?: string;
  patient_id?: string;
  constraints?: Record<string, unknown>;
  top_k?: number;
};

async function post(path: string, payload: ChatPayload) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export async function runChat(pipeline: Pipeline, payload: ChatPayload) {
  if (pipeline === "naive") {
    return post("/naive/chat", payload);
  }
  if (pipeline === "both") {
    return post("/naive/compare", payload);
  }
  return post("/medswin/chat", payload);
}

export async function getTrace(traceId: string, orgId: string) {
  const url = new URL(`${API_BASE}/medswin/traces/${traceId}`, window.location.origin);
  url.searchParams.set("org_id", orgId);
  url.searchParams.set("include_details", "true");
  const res = await fetch(url.toString().replace(window.location.origin, ""));
  if (!res.ok) {
    throw new Error(`Trace fetch failed (${res.status})`);
  }
  return res.json();
}

export function portalUrls(evalPort = 8200) {
  const origin = window.location.origin;
  const evalOrigin = `${window.location.protocol}//${window.location.hostname}:${evalPort}`;
  return {
    clinician: `${origin}/app/`,
    dashboard: `${origin}/api/v1/dashboard/`,
    docs: `${origin}/docs`,
    eval: `${evalOrigin}/`,
  };
}
