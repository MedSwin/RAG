const API_BASE = import.meta.env.VITE_API_BASE || "/api/v1";

export type ChatPayload = {
  query: string;
  user_id: string;
  org_id: string;
  session_id?: string;
  patient_id?: string;
  constraints?: Record<string, unknown>;
};

export async function chat(payload: ChatPayload) {
  const res = await fetch(`${API_BASE}/medswin/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `Chat failed (${res.status})`);
  }
  return res.json();
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
