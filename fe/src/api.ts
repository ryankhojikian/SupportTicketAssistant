/** Mirrors `be/schemas.py` AnalyzeResponse and metrics/logs shapes. */

export interface MethodResult {
  label: string | null;
  confidence: number | null;
  raw_output: string | null;
  latency_ms: number;
  cost_usd: number;
  error: string | null;
}

export interface RagRetrieval {
  documents: string[];
  metadatas: Record<string, unknown>[];
  distances: number[];
  ids: string[];
}

export interface SummaryRow {
  system: string;
  priority: string | null;
  confidence: number | null;
  latency_ms: number;
  cost_usd: number;
  error: string | null;
}

export interface AnalyzeResponse {
  tweet: string;
  rag_retrieval: RagRetrieval;
  methods: {
    ml: MethodResult;
    llm_zero_shot: MethodResult;
    llm_non_rag: MethodResult;
    llm_rag: MethodResult;
  };
  summary_table: SummaryRow[];
}

// ── Metrics ──────────────────────────────────────────────────────────────────

export interface SystemMetrics {
  calls: number;
  errors: number;
  avg_latency_ms: number;
  total_latency_ms: number;
  total_cost_usd: number;
  label_counts: { Urgent: number; Normal: number };
}

export interface RecentQuery {
  ts: string;
  query_snippet: string;
  total_latency_ms: number;
  had_errors: boolean;
  labels: Record<string, string | null>;
}

export interface MetricsResponse {
  total_queries: number;
  had_errors_count: number;
  systems: Record<string, SystemMetrics>;
  recent_queries: RecentQuery[];
}

// ── Log entries ───────────────────────────────────────────────────────────────

export interface LogEntry {
  ts?: string;
  level?: string;
  msg?: string;
  event?: string;
  [key: string]: unknown;
  raw?: string;
}

export interface LogsResponse {
  entries: LogEntry[];
  total_lines: number;
  log_file: string;
}

// ── URL helpers ───────────────────────────────────────────────────────────────

function apiUrl(path: string): string {
  const base = import.meta.env.VITE_API_BASE?.replace(/\/$/, "") ?? "";
  if (base) return `${base}${path}`;
  return `/api${path}`;
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(apiUrl(path), init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = (await res.json()) as { detail?: string };
      if (j.detail) detail = j.detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ── Public API ────────────────────────────────────────────────────────────────

export async function analyzeTweet(tweet: string): Promise<AnalyzeResponse> {
  return apiFetch<AnalyzeResponse>("/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tweet }),
  });
}

export async function fetchMetrics(): Promise<MetricsResponse> {
  return apiFetch<MetricsResponse>("/metrics");
}

export async function fetchLogs(limit = 50): Promise<LogsResponse> {
  return apiFetch<LogsResponse>(`/logs?limit=${limit}`);
}
