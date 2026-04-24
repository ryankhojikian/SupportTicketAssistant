import React, { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import {
  analyzeTweet,
  fetchLogs,
  fetchMetrics,
  type AnalyzeResponse,
  type LogEntry,
  type MetricsResponse,
  type RecentQuery,
  type SummaryRow,
  type SystemMetrics,
} from "./api";

// ── Helpers ───────────────────────────────────────────────────────────────────

function formatUsd(n: number): string {
  if (n === 0) return "$0";
  if (n < 0.0001) return `<$0.0001`;
  return `$${n.toFixed(6)}`;
}

function formatMs(ms: number): string {
  return ms < 1 ? "<1 ms" : `${ms.toFixed(1)} ms`;
}

function relativeTime(isoString: string): string {
  const diff = (Date.now() - new Date(isoString).getTime()) / 1000;
  if (diff < 60) return `${Math.round(diff)}s ago`;
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  return `${Math.round(diff / 3600)}h ago`;
}

const SYSTEM_LABELS: Record<string, string> = {
  "ML (Random Forest)": "ML",
  llm_zero_shot: "Zero-shot",
  llm_non_rag: "Non-RAG",
  llm_rag: "RAG",
};

// ── Sub-components ────────────────────────────────────────────────────────────

function RetrievedSources({ data }: { data: AnalyzeResponse["rag_retrieval"] }) {
  const { documents, distances, metadatas, ids } = data;
  const n = Math.max(documents.length, distances.length, metadatas.length, ids.length);
  if (n === 0) {
    return (
      <p className="muted empty-hint">No similar tickets returned from the vector store.</p>
    );
  }
  const rows: { doc: string; dist?: number; meta?: Record<string, unknown>; id?: string }[] = [];
  for (let i = 0; i < n; i++) {
    rows.push({ doc: documents[i] ?? "", dist: distances[i], meta: metadatas[i], id: ids[i] });
  }
  return (
    <ul className="source-list">
      {rows.map((r, i) => (
        <li key={r.id ?? i} className="source-card">
          <div className="source-card-head">
            <span className="badge">#{i + 1}</span>
            {r.dist !== undefined && (
              <span className="mono faint" title="Lower is more similar (distance)">
                distance {r.dist.toFixed(4)}
              </span>
            )}
            {r.id && <span className="mono faint">id {r.id}</span>}
          </div>
          <p className="source-body">{r.doc || "(empty)"}</p>
          {r.meta && Object.keys(r.meta).length > 0 && (
            <pre className="meta-block">{JSON.stringify(r.meta, null, 2)}</pre>
          )}
        </li>
      ))}
    </ul>
  );
}

function AnswerPanel({
  title,
  subtitle,
  text,
  error,
}: {
  title: string;
  subtitle?: string;
  text: string | null;
  error: string | null;
}) {
  return (
    <article className="answer-panel">
      <header>
        <h3>{title}</h3>
        {subtitle && <p className="muted small">{subtitle}</p>}
      </header>
      {error ? (
        <p className="error-inline">{error}</p>
      ) : (
        <div className="answer-body">{text?.trim() || "—"}</div>
      )}
    </article>
  );
}

// ── Dashboard sub-components ──────────────────────────────────────────────────

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="stat-card">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
      {sub && <span className="stat-sub muted small">{sub}</span>}
    </div>
  );
}

function LatencyBar({ ms, max }: { ms: number; max: number }) {
  const pct = max > 0 ? Math.min((ms / max) * 100, 100) : 0;
  return (
    <div className="latency-bar-wrap" title={`${ms.toFixed(1)} ms`}>
      <div className="latency-bar" style={{ width: `${pct}%` }} />
      <span className="latency-bar-label mono">{formatMs(ms)}</span>
    </div>
  );
}

function SystemsTable({ metrics }: { metrics: MetricsResponse }) {
  const systems: [string, SystemMetrics][] = Object.entries(metrics.systems);
  const maxLatency = Math.max(...systems.map(([, s]) => s.avg_latency_ms), 1);
  const maxCost = Math.max(...systems.map(([, s]) => s.total_cost_usd), 0.000001);

  return (
    <div className="table-wrap">
      <table className="compare-table metrics-table">
        <thead>
          <tr>
            <th>System</th>
            <th>Calls</th>
            <th>Errors</th>
            <th>Avg Latency</th>
            <th>Total Cost</th>
            <th>Urgent</th>
            <th>Normal</th>
          </tr>
        </thead>
        <tbody>
          {systems.map(([name, s]) => {
            const costPct = maxCost > 0 ? Math.min((s.total_cost_usd / maxCost) * 100, 100) : 0;
            return (
              <tr key={name}>
                <td>
                  <span className="chip-label">{SYSTEM_LABELS[name] ?? name}</span>
                </td>
                <td className="mono">{s.calls}</td>
                <td className={s.errors > 0 ? "error-cell mono" : "mono"}>{s.errors}</td>
                <td style={{ minWidth: 160 }}>
                  <LatencyBar ms={s.avg_latency_ms} max={maxLatency} />
                </td>
                <td style={{ minWidth: 120 }}>
                  <div className="latency-bar-wrap" title={`$${s.total_cost_usd.toFixed(6)}`}>
                    <div
                      className="latency-bar cost-bar"
                      style={{ width: `${costPct}%` }}
                    />
                    <span className="latency-bar-label mono">{formatUsd(s.total_cost_usd)}</span>
                  </div>
                </td>
                <td className="mono urgent-cell">{s.label_counts.Urgent}</td>
                <td className="mono">{s.label_counts.Normal}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function RecentQueriesList({ metrics }: { metrics: MetricsResponse }) {
  if (metrics.recent_queries.length === 0) {
    return <p className="muted empty-hint">No queries yet.</p>;
  }
  return (
    <ul className="recent-list">
      {metrics.recent_queries.map((q, i) => (
        <li key={i} className="recent-item">
          <div className="recent-head">
            <span className="recent-time muted small">{relativeTime(q.ts)}</span>
            <span className="recent-latency mono small">{formatMs(q.total_latency_ms)}</span>
            {q.had_errors && <span className="badge-error">error</span>}
          </div>
          <p className="recent-query">{q.query_snippet}</p>
          <div className="chips-row small-chips">
            {(Object.entries(q.labels) as [string, string | null][]).map(([sys, lbl]) => (
              <span
                key={sys}
                className={`chip chip-sm ${lbl === "Urgent" ? "chip-urgent" : ""}`}
              >
                <span className="chip-label">{SYSTEM_LABELS[sys] ?? sys}</span>{" "}
                {lbl ?? "—"}
              </span>
            ))}
          </div>
        </li>
      ))}
    </ul>
  );
}

function LogViewer({
  entries,
  totalLines,
  logFile,
}: {
  entries: LogEntry[];
  totalLines: number;
  logFile: string;
}) {
  if (entries.length === 0) {
    return <p className="muted empty-hint">No log entries yet.</p>;
  }

  function levelClass(level?: string): string {
    if (!level) return "";
    if (level === "ERROR") return "log-level-error";
    if (level === "WARNING") return "log-level-warn";
    if (level === "DEBUG") return "log-level-debug";
    return "log-level-info";
  }

  return (
    <>
      <p className="muted small log-meta">
        Showing last {entries.length} of {totalLines} lines ·{" "}
        <code className="mono">{logFile}</code>
      </p>
      <div className="log-scroll">
        {entries.map((e, i) => (
          <div key={i} className={`log-row ${levelClass(e.level)}`}>
            <span className="log-ts mono">{e.ts ? e.ts.replace("T", " ").replace(/\.\d+Z$/, " UTC") : ""}</span>
            <span className={`log-badge ${levelClass(e.level)}`}>{e.level ?? "LOG"}</span>
            {e.event && <span className="log-event">{e.event}</span>}
            <span className="log-msg">{e.raw ?? e.msg ?? ""}</span>
          </div>
        ))}
      </div>
    </>
  );
}

// ── Dashboard panel ───────────────────────────────────────────────────────────

function Dashboard() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [logs, setLogs] = useState<{ entries: LogEntry[]; totalLines: number; logFile: string } | null>(null);
  const [loadingMetrics, setLoadingMetrics] = useState(false);
  const [loadingLogs, setLoadingLogs] = useState(false);
  const [metricsErr, setMetricsErr] = useState<string | null>(null);
  const [logsErr, setLogsErr] = useState<string | null>(null);
  const [logLimit, setLogLimit] = useState(50);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const refresh = useCallback(async () => {
    setLoadingMetrics(true);
    setMetricsErr(null);
    try {
      const m = await fetchMetrics();
      setMetrics(m);
    } catch (e) {
      setMetricsErr(e instanceof Error ? e.message : "Failed to load metrics");
    } finally {
      setLoadingMetrics(false);
    }
  }, []);

  const refreshLogs = useCallback(async (limit: number) => {
    setLoadingLogs(true);
    setLogsErr(null);
    try {
      const l = await fetchLogs(limit);
      setLogs({ entries: l.entries, totalLines: l.total_lines, logFile: l.log_file });
    } catch (e) {
      setLogsErr(e instanceof Error ? e.message : "Failed to load logs");
    } finally {
      setLoadingLogs(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    refreshLogs(logLimit);
    intervalRef.current = setInterval(() => {
      refresh();
      refreshLogs(logLimit);
    }, 10_000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [refresh, refreshLogs, logLimit]);

  const totalCost =
    metrics
      ? Object.values(metrics.systems).reduce((s, sys) => s + sys.total_cost_usd, 0)
      : 0;
  const avgTotalLatency =
    metrics && metrics.total_queries > 0
      ? metrics.recent_queries.reduce((s: number, q: RecentQuery) => s + q.total_latency_ms, 0) /
        metrics.recent_queries.length
      : 0;

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h2>Live Dashboard</h2>
        <div className="dashboard-actions">
          <button
            className="btn-ghost"
            onClick={() => { refresh(); refreshLogs(logLimit); }}
            disabled={loadingMetrics || loadingLogs}
          >
            {loadingMetrics || loadingLogs ? "Refreshing…" : "↻ Refresh"}
          </button>
          <span className="muted small">Auto-refresh every 10 s</span>
        </div>
      </div>

      {/* Stat cards */}
      {metrics && (
        <div className="stat-grid">
          <StatCard
            label="Total queries"
            value={String(metrics.total_queries)}
            sub={`${metrics.had_errors_count} with errors`}
          />
          <StatCard
            label="Avg round-trip"
            value={formatMs(avgTotalLatency)}
            sub="wall-clock per query"
          />
          <StatCard
            label="Cumulative LLM cost"
            value={formatUsd(totalCost)}
            sub="all systems, all calls"
          />
          <StatCard
            label="Urgent / Normal"
            value={(() => {
              const ml = metrics.systems["ML (Random Forest)"];
              if (!ml) return "—";
              return `${ml.label_counts.Urgent} / ${ml.label_counts.Normal}`;
            })()}
            sub="ML predictions"
          />
        </div>
      )}

      {metricsErr && <p className="error-inline">{metricsErr}</p>}

      {/* Per-system breakdown */}
      {metrics && (
        <section className="section">
          <h3>Per-system breakdown</h3>
          <p className="muted small section-desc">
            Latency bars are relative to the slowest system. Cost bars are relative to the
            highest cumulative spend.
          </p>
          <SystemsTable metrics={metrics} />
        </section>
      )}

      {/* Recent queries */}
      {metrics && (
        <section className="section">
          <h3>Recent queries <span className="muted small">(last {metrics.recent_queries.length})</span></h3>
          <RecentQueriesList metrics={metrics} />
        </section>
      )}

      {/* Log viewer */}
      <section className="section">
        <div className="log-viewer-head">
          <h3>Structured logs</h3>
          <select
            className="log-limit-select"
            value={logLimit}
            onChange={(e: React.ChangeEvent<HTMLSelectElement>) => {
              const v = Number(e.target.value);
              setLogLimit(v);
              refreshLogs(v);
            }}
          >
            {[25, 50, 100, 200].map((n) => (
              <option key={n} value={n}>{n} lines</option>
            ))}
          </select>
        </div>
        {logsErr && <p className="error-inline">{logsErr}</p>}
        {logs && (
          <LogViewer entries={logs.entries} totalLines={logs.totalLines} logFile={logs.logFile} />
        )}
      </section>
    </div>
  );
}

// ── Root App ──────────────────────────────────────────────────────────────────

type Tab = "analyze" | "dashboard";

export default function App() {
  const [tab, setTab] = useState<Tab>("analyze");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);

  const onSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      const tweet = query.trim();
      if (!tweet) return;
      setLoading(true);
      setError(null);
      try {
        const data = await analyzeTweet(tweet);
        setResult(data);
      } catch (err) {
        setResult(null);
        setError(err instanceof Error ? err.message : "Request failed");
      } finally {
        setLoading(false);
      }
    },
    [query],
  );

  const m = result?.methods;

  return (
    <div className="app">
      <header className="hero">
        <p className="eyebrow">Week 3 · Decision Intelligence</p>
        <h1>Support ticket assistant</h1>
        <p className="lede">
          Ask a question as if it were a customer tweet. The backend runs RAG retrieval, parallel
          LLM paths, and an ML priority baseline — compare answers, sources, latency, and cost.
        </p>
      </header>

      {/* Tab bar */}
      <nav className="tab-bar" role="tablist">
        <button
          role="tab"
          aria-selected={tab === "analyze"}
          className={tab === "analyze" ? "tab active" : "tab"}
          onClick={() => setTab("analyze")}
        >
          Analyze
        </button>
        <button
          role="tab"
          aria-selected={tab === "dashboard"}
          className={tab === "dashboard" ? "tab active" : "tab"}
          onClick={() => setTab("dashboard")}
        >
          Dashboard &amp; Logs
        </button>
      </nav>

      {tab === "analyze" && (
        <>
          <form className="query-form" onSubmit={onSubmit}>
            <label htmlFor="tweet">Your ticket / query</label>
            <textarea
              id="tweet"
              name="tweet"
              rows={4}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. My package is 3 days late and I need a refund immediately!!"
              disabled={loading}
            />
            <div className="form-actions">
              <button type="submit" disabled={loading || !query.trim()}>
                {loading ? "Analyzing…" : "Analyze"}
              </button>
            </div>
          </form>

          {error && (
            <div className="banner error" role="alert">
              {error}
            </div>
          )}

          {result && (
            <>
              <section className="section">
                <h2>Retrieved tickets</h2>
                <p className="muted section-desc">
                  Top similar cases from the vector store (used as RAG context).
                </p>
                <RetrievedSources data={result.rag_retrieval} />
              </section>

              <section className="section">
                <h2>Answers</h2>
                <p className="muted section-desc">
                  Non-RAG vs RAG LLM replies (full model output). Zero-shot path only returns a
                  priority label; see the comparison table below.
                </p>
                <div className="answer-grid">
                  <AnswerPanel
                    title="LLM without RAG"
                    subtitle="General knowledge only"
                    text={m?.llm_non_rag.raw_output ?? null}
                    error={m?.llm_non_rag.error ?? null}
                  />
                  <AnswerPanel
                    title="LLM with RAG"
                    subtitle="Grounded on retrieved tickets"
                    text={m?.llm_rag.raw_output ?? null}
                    error={m?.llm_rag.error ?? null}
                  />
                </div>
              </section>

              <section className="section">
                <h2>Four-way comparison</h2>
                <p className="muted section-desc">
                  Per-call latency and cost come from this request. ML test-set accuracy is saved
                  when training runs and loaded at startup. LLM confidence is self-reported by the
                  model per call; LLMs have no offline test-set evaluation.
                </p>
                <div className="table-wrap">
                  <table className="compare-table">
                    <thead>
                      <tr>
                        <th>System</th>
                        <th>Priority</th>
                        <th>Confidence</th>
                        <th>Latency (ms)</th>
                        <th>Cost (USD)</th>
                        <th>Test accuracy</th>
                        <th>Error</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.summary_table.map((row: SummaryRow) => (
                        <tr key={row.system}>
                          <td>{row.system}</td>
                          <td>{row.priority ?? "—"}</td>
                          <td>
                            {row.confidence != null
                              ? `${(row.confidence * 100).toFixed(1)}%`
                              : "—"}
                          </td>
                          <td className="mono">{row.latency_ms.toFixed(1)}</td>
                          <td className="mono">{formatUsd(row.cost_usd)}</td>
                          <td className="mono">
                            {row.test_accuracy != null
                              ? `${(row.test_accuracy * 100).toFixed(1)}%`
                              : "—"}
                          </td>
                          <td className="error-cell">{row.error ?? "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>

              <section className="section">
                <h2>Raw priority outputs</h2>
                <div className="chips-row">
                  <span className="chip">
                    <span className="chip-label">ML</span> {m?.ml.label ?? "—"}
                    {m?.ml.confidence != null && (
                      <span className="faint"> ({(m.ml.confidence * 100).toFixed(0)}%)</span>
                    )}
                  </span>
                  <span className="chip">
                    <span className="chip-label">LLM zero-shot</span>{" "}
                    {m?.llm_zero_shot.label ?? "—"}
                  </span>
                </div>
                {(m?.llm_zero_shot.error || m?.ml.error) && (
                  <p className="muted small">
                    {m?.ml.error && <>ML: {m.ml.error} </>}
                    {m?.llm_zero_shot.error && <>Zero-shot: {m.llm_zero_shot.error}</>}
                  </p>
                )}
              </section>
            </>
          )}
        </>
      )}

      {tab === "dashboard" && <Dashboard />}

      <footer className="footer muted small">
        API: <code className="mono">POST /analyze</code> ·{" "}
        <code className="mono">GET /metrics</code> ·{" "}
        <code className="mono">GET /logs?limit=N</code>. Dev server proxies{" "}
        <code className="mono">/api/*</code> → FastAPI (override with{" "}
        <code className="mono">VITE_API_BASE</code>).
      </footer>
    </div>
  );
}
