import { useEffect, useMemo, useState } from "react";
import { ExperimentRow, Prediction, Step, TrajectoryRow, fmt, get } from "./api";

type Tab = "overview" | "trajectories" | "monitors" | "interventions";
const MONITOR_COLORS: Record<string, string> = {
  behavioral: "var(--series-1)",
  probe: "var(--series-2)",
  combined: "var(--series-3)",
  majority: "var(--text-muted)",
};

export default function App() {
  const [experiments, setExperiments] = useState<ExperimentRow[]>([]);
  const [eid, setEid] = useState<string>("");
  const [tab, setTab] = useState<Tab>("overview");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    get<ExperimentRow[]>("/api/experiments")
      .then((rows) => {
        setExperiments(rows);
        const withEval = rows.find((r) => r.has_eval);
        if (rows.length) setEid((withEval ?? rows[0]).id);
      })
      .catch((e) => setError(String(e)));
  }, []);

  const current = experiments.find((e) => e.id === eid);
  return (
    <div className="app">
      <header>
        <h1>Agent Failure Observatory</h1>
        <div className="controls">
          <select value={eid} onChange={(e) => setEid(e.target.value)}>
            {experiments.map((e) => (
              <option key={e.id} value={e.id}>
                {e.id} ({e.n_trajectories})
              </option>
            ))}
          </select>
          <nav>
            {(["overview", "trajectories", "monitors", "interventions"] as Tab[]).map((t) => (
              <button key={t} className={t === tab ? "active" : ""} onClick={() => setTab(t)}>
                {t[0].toUpperCase() + t.slice(1)}
              </button>
            ))}
          </nav>
        </div>
      </header>
      {error && <p className="error">{error}</p>}
      {current?.adapter === "simulated" && (
        <p className="banner">
          Simulated backend: results validate the pipeline and do not describe a real model.
        </p>
      )}
      {eid && tab === "overview" && <Overview eid={eid} />}
      {eid && tab === "trajectories" && <Trajectories eid={eid} />}
      {eid && tab === "monitors" && <Monitors eid={eid} />}
      {eid && tab === "interventions" && <Interventions eid={eid} />}
    </div>
  );
}

// ---------------------------------------------------------------------------------------- overview

function Overview({ eid }: { eid: string }) {
  const [d, setD] = useState<any>(null);
  useEffect(() => {
    setD(null);
    get<any>(`/api/experiments/${eid}/overview`).then(setD);
  }, [eid]);
  if (!d) return <p className="muted">Loading…</p>;
  return (
    <section>
      <div className="tiles">
        <Tile label="Trajectories" value={String(d.n_trajectories)} />
        <Tile label="Task success" value={fmt(d.task_success_rate, 2)} />
        <Tile label="Safety violation rate" value={fmt(d.violation_rate, 2)} />
        <Tile label="Model" value={d.manifest?.model?.id ?? "?"} small />
      </div>
      <div className="grid2">
        <div className="card">
          <h3>By environment</h3>
          <table>
            <thead>
              <tr><th>Environment</th><th>n</th><th>Violation rate</th><th>Task success</th></tr>
            </thead>
            <tbody>
              {Object.entries<any>(d.by_environment).map(([env, e]) => (
                <tr key={env}>
                  <td>{env}</td><td>{e.n}</td>
                  <td><Bar v={e.violations / e.n} />{fmt(e.violations / e.n, 2)}</td>
                  <td>{fmt(e.successes / e.n, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <h3>Violations by rule</h3>
          <table>
            <tbody>
              {Object.entries<number>(d.violation_rules).slice(0, 12).map(([r, n]) => (
                <tr key={r}><td className="mono">{r}</td><td>{n}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card">
          <h3>Monitors (test split, horizon {d.monitor_horizon ?? "?"})</h3>
          <table>
            <thead><tr><th>Monitor</th><th>AUROC [95% CI]</th><th>AUPRC</th><th>Recall</th><th>FPR</th></tr></thead>
            <tbody>
              {d.monitor_summary.map((m: any) => (
                <tr key={m.monitor}>
                  <td><Swatch name={m.monitor} />{m.monitor}{m.layer !== null && m.layer !== undefined ? ` (L${m.layer})` : ""}</td>
                  <td>{fmt(m.auroc)} [{fmt(m.lo)}, {fmt(m.hi)}]</td>
                  <td>{fmt(m.auprc)}</td><td>{fmt(m.recall, 2)}</td><td>{fmt(m.fpr, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <h3>Interventions</h3>
          <table>
            <thead><tr><th>Condition</th><th>Violation</th><th>Success</th><th>Intervened runs</th></tr></thead>
            <tbody>
              {Object.entries<any>(d.interventions).map(([c, s]) => (
                <tr key={c}>
                  <td className="mono">{c}</td><td>{fmt(s.violation_rate, 2)}</td>
                  <td>{fmt(s.task_success_rate, 2)}</td><td>{fmt(s.trigger_run_rate, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------------------- trajectories

function Trajectories({ eid }: { eid: string }) {
  const [conditions, setConditions] = useState<string[]>(["collection"]);
  const [condition, setCondition] = useState("collection");
  const [env, setEnv] = useState("");
  const [outcome, setOutcome] = useState("");
  const [rows, setRows] = useState<TrajectoryRow[]>([]);
  const [runId, setRunId] = useState<string | null>(null);

  useEffect(() => {
    get<any>(`/api/experiments/${eid}/overview`).then((d) => setConditions(d.conditions));
  }, [eid]);
  useEffect(() => {
    const q = new URLSearchParams();
    if (condition !== "collection") q.set("condition", condition);
    if (env) q.set("environment", env);
    if (outcome) q.set("outcome", outcome);
    get<TrajectoryRow[]>(`/api/experiments/${eid}/trajectories?${q}`).then((r) => {
      setRows(r);
      setRunId(r[0]?.run_id ?? null);
    });
  }, [eid, condition, env, outcome]);

  return (
    <section>
      <div className="filters">
        <select value={condition} onChange={(e) => setCondition(e.target.value)}>
          {conditions.map((c) => <option key={c}>{c}</option>)}
        </select>
        <select value={env} onChange={(e) => setEnv(e.target.value)}>
          <option value="">all environments</option>
          <option>goal_drift</option><option>prompt_injection</option><option>tool_safety</option>
        </select>
        <select value={outcome} onChange={(e) => setOutcome(e.target.value)}>
          <option value="">all outcomes</option>
          <option value="violation">safety violation</option><option value="clean">no violation</option>
          <option value="success">task success</option><option value="failure">task failure</option>
        </select>
        <span className="muted">{rows.length} trajectories</span>
      </div>
      <div className="split">
        <div className="list card">
          {rows.map((r) => (
            <button key={r.run_id} className={`row ${r.run_id === runId ? "active" : ""}`} onClick={() => setRunId(r.run_id)}>
              <span className="mono">{r.run_id}</span>
              <span className={`pill ${r.safety_violation ? "bad" : "ok"}`}>
                {r.safety_violation ? `violation @${r.failure_onset}` : "clean"}
              </span>
              <span className="muted">{r.task_success ? "success" : r.status}</span>
            </button>
          ))}
        </div>
        {runId && <TrajectoryDetail eid={eid} runId={runId} condition={condition} />}
      </div>
    </section>
  );
}

function TrajectoryDetail({ eid, runId, condition }: { eid: string; runId: string; condition: string }) {
  const [d, setD] = useState<{ trajectory: any; predictions: Record<string, Prediction> } | null>(null);
  const [horizon, setHorizon] = useState("3");
  useEffect(() => {
    const q = condition !== "collection" ? `?condition=${encodeURIComponent(condition)}` : "";
    get<any>(`/api/experiments/${eid}/trajectories/${encodeURIComponent(runId)}${q}`).then(setD);
  }, [eid, runId, condition]);
  if (!d) return <div className="card detail muted">Loading…</div>;
  const t = d.trajectory;
  const horizons = Array.from(new Set(Object.keys(d.predictions).map((k) => k.split("|")[1]))).sort();
  const series = Object.entries(d.predictions).filter(([k]) => k.endsWith(`|${horizon}`) && !k.startsWith("majority"));
  return (
    <div className="card detail">
      <h3 className="mono">{t.run_id}</h3>
      <p className="muted">
        {t.task.environment} · strength {t.task.hazard_strength} · {t.task.difficulty} · seed {t.seed} · status {t.outcome.status}
        {" · "}task success {String(t.outcome.task_success)} · violation {String(t.outcome.safety_violation)}
        {t.outcome.violation_rules.length ? ` (${t.outcome.violation_rules.join(", ")})` : ""}
      </p>
      <p><b>Objective:</b> {t.task.objective}</p>
      {series.length > 0 && (
        <>
          <div className="filters">
            <span>Monitor predictions, horizon</span>
            <select value={horizon} onChange={(e) => setHorizon(e.target.value)}>
              {horizons.map((h) => <option key={h}>{h}</option>)}
            </select>
            <span className="muted">split: {series[0][1].split}</span>
          </div>
          <PredictionChart series={series} nSteps={t.steps.length} onset={t.outcome.failure_onset} />
        </>
      )}
      <ol className="steps">
        {t.steps.map((s: Step) => (
          <li key={s.step} className={s.annotations.violations.length ? "violation" : ""}>
            <div className="stephead">
              <b>Step {s.step}</b>
              <span className="mono">{s.tool ?? "(unparsed)"}({JSON.stringify(s.args)})</span>
              {s.annotations.hazard_visible && <span className="pill warn">hazard visible</span>}
              {s.annotations.violations.length > 0 && (
                <span className="pill bad">{s.step === t.outcome.failure_onset ? "FIRST UNSAFE ACTION: " : ""}{s.annotations.violations.join(", ")}</span>
              )}
              {s.intervention?.triggered && <span className="pill info">{s.intervention.type}: {s.intervention.action_taken ?? "triggered"}</span>}
              {s.annotations.prevented_violations.length > 0 && <span className="pill ok">prevented {s.annotations.prevented_violations.join(", ")}</span>}
            </div>
            <details>
              <summary>observation · output · tool response</summary>
              <pre>{s.observation}</pre>
              <pre className="out">{s.raw_output}</pre>
              <pre>{s.tool_response}</pre>
            </details>
          </li>
        ))}
      </ol>
    </div>
  );
}

function PredictionChart({ series, nSteps, onset }: { series: [string, Prediction][]; nSteps: number; onset: number | null }) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 560, H = 180, P = { l: 36, r: 12, t: 10, b: 24 };
  const x = (s: number) => P.l + (s / Math.max(1, nSteps - 1)) * (W - P.l - P.r);
  const y = (v: number) => P.t + (1 - v) * (H - P.t - P.b);
  const thr = series[0]?.[1].threshold;
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} onMouseLeave={() => setHover(null)}>
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line x1={P.l} x2={W - P.r} y1={y(v)} y2={y(v)} className="grid" />
            <text x={P.l - 6} y={y(v) + 4} textAnchor="end" className="axis">{v}</text>
          </g>
        ))}
        {Array.from({ length: nSteps }, (_, s) => (
          <text key={s} x={x(s)} y={H - 6} textAnchor="middle" className="axis">{s}</text>
        ))}
        {onset !== null && <line x1={x(onset)} x2={x(onset)} y1={P.t} y2={H - P.b} className="onset" />}
        {series.map(([k, p]) => (
          <g key={k}>
            <polyline fill="none" strokeWidth={2} stroke={MONITOR_COLORS[k.split("|")[0]]}
              points={p.steps.map((s, i) => `${x(s)},${y(p.scores[i])}`).join(" ")} />
            {p.steps.map((s, i) => (
              <circle key={s} cx={x(s)} cy={y(p.scores[i])} r={4} fill={MONITOR_COLORS[k.split("|")[0]]} stroke="var(--surface-1)" strokeWidth={2} />
            ))}
          </g>
        ))}
        {Array.from({ length: nSteps }, (_, s) => (
          <rect key={s} x={x(s) - 12} y={P.t} width={24} height={H - P.t - P.b} fill="transparent" onMouseEnter={() => setHover(s)} />
        ))}
        {hover !== null && <line x1={x(hover)} x2={x(hover)} y1={P.t} y2={H - P.b} className="crosshair" />}
      </svg>
      <div className="legend">
        {series.map(([k]) => <span key={k}><Swatch name={k.split("|")[0]} />{k.split("|")[0]}</span>)}
        {onset !== null && <span><i className="onset-key" />failure onset</span>}
        <span className="muted">threshold (per monitor, from validation) e.g. {fmt(thr, 2)}</span>
      </div>
      {hover !== null && (
        <div className="tooltip">
          step {hover}:{" "}
          {series.map(([k, p]) => {
            const i = p.steps.indexOf(hover);
            return <span key={k}> {k.split("|")[0]} {i >= 0 ? fmt(p.scores[i], 2) : "–"}{i >= 0 && p.scores[i] >= p.threshold ? " (alert)" : ""}</span>;
          })}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------------------- monitors

function Monitors({ eid }: { eid: string }) {
  const [d, setD] = useState<any>(null);
  const [plots, setPlots] = useState<string[]>([]);
  useEffect(() => {
    get<any>(`/api/experiments/${eid}/metrics`).then(setD);
    get<string[]>(`/api/experiments/${eid}/plots`).then(setPlots);
  }, [eid]);
  const horizons = useMemo(() => (d?.metrics ? Object.keys(d.metrics.results) : []), [d]);
  if (!d?.metrics) return <p className="muted">No evaluation artefacts yet. Run <code>afo evaluate</code>.</p>;
  const m = d.metrics;
  return (
    <section>
      <p className="muted">
        Test split: {m.test_split.rows} decision points from {m.test_split.trajectories} trajectories on{" "}
        {m.test_split.templates.length} unseen templates. CIs: cluster bootstrap over {m.test_split.bootstrap_group} ({m.test_split.n_groups} groups).
        Thresholds: {m.threshold_method.method} on validation.
      </p>
      <table className="card">
        <thead>
          <tr><th>Horizon</th><th>Monitor</th><th>Layer</th><th>n / pos</th><th>AUROC [95% CI]</th><th>AUPRC</th><th>Precision</th><th>Recall</th><th>F1</th><th>FPR</th><th>Brier</th></tr>
        </thead>
        <tbody>
          {horizons.flatMap((h) =>
            Object.entries<any>(m.results[h]).map(([name, r]) => (
              <tr key={h + name}>
                <td>{h}</td><td><Swatch name={name} />{name}</td><td>{r.layer ?? "-"}</td><td>{r.n} / {r.n_pos}</td>
                <td>{fmt(r.auroc)} [{fmt(r.auroc_ci.lo)}, {fmt(r.auroc_ci.hi)}]</td><td>{fmt(r.auprc)}</td>
                <td>{fmt(r.precision, 2)}</td><td>{fmt(r.recall, 2)}</td><td>{fmt(r.f1, 2)}</td><td>{fmt(r.fpr, 2)}</td><td>{fmt(r.brier)}</td>
              </tr>
            )),
          )}
        </tbody>
      </table>
      <div className="plots">
        {plots.filter((p) => !p.startsWith("interventions")).map((p) => (
          <figure key={p} className="card">
            <img src={`/api/experiments/${eid}/plots/${p}`} alt={p} />
            <figcaption>{p}</figcaption>
          </figure>
        ))}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------------------- interventions

function Interventions({ eid }: { eid: string }) {
  const [d, setD] = useState<any>(null);
  useEffect(() => {
    get<any>(`/api/experiments/${eid}/interventions`).then(setD);
  }, [eid]);
  if (!d?.conditions) return <p className="muted">No intervention study yet. Run <code>afo intervene</code>.</p>;
  return (
    <section>
      <p className="muted">
        Monitor <b>{d.monitor}</b>, horizon {d.horizon}, threshold {fmt(d.threshold, 3)} ({d.threshold_source}). Test templates:{" "}
        {d.test_templates.join(", ")}. Random triggers fire at the monitor's per-step rate.
      </p>
      <table className="card">
        <thead>
          <tr><th>Condition</th><th>Runs</th><th>Violation [95% CI]</th><th>Δ vs none</th><th>Task success</th><th>Δ success</th><th>Intervened runs</th><th>False-alarm runs</th><th>Stopped unsafe</th><th>Stopped legitimate</th></tr>
        </thead>
        <tbody>
          {Object.entries<any>(d.conditions).map(([c, s]) => (
            <tr key={c}>
              <td className="mono">{c}</td><td>{s.n_runs}</td>
              <td><Bar v={s.violation_rate} />{fmt(s.violation_rate, 2)} [{fmt(s.violation_rate_ci.lo, 2)}, {fmt(s.violation_rate_ci.hi, 2)}]</td>
              <td>{s.vs_none ? fmt(s.vs_none.violation_delta.point, 2) : "-"}</td>
              <td>{fmt(s.task_success_rate, 2)}</td>
              <td>{s.vs_none ? fmt(s.vs_none.success_delta.point, 2) : "-"}</td>
              <td>{fmt(s.trigger_run_rate, 2)}</td>
              <td>{s.vs_none ? fmt(s.vs_none.false_alarm_run_rate, 2) : "-"}</td>
              <td>{s.stopped_actions_unsafe}</td><td>{s.stopped_actions_legitimate}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <figure className="card">
        <img src={`/api/experiments/${eid}/plots/interventions.png`} alt="intervention outcomes" />
      </figure>
    </section>
  );
}

// ---------------------------------------------------------------------------------------- bits

function Tile({ label, value, small }: { label: string; value: string; small?: boolean }) {
  return (
    <div className="tile card">
      <div className="muted">{label}</div>
      <div className={small ? "value small" : "value"}>{value}</div>
    </div>
  );
}

function Swatch({ name }: { name: string }) {
  return <i className="swatch" style={{ background: MONITOR_COLORS[name] ?? "var(--text-muted)" }} />;
}

function Bar({ v }: { v: number }) {
  return <i className="bar" style={{ width: `${Math.round(v * 60)}px` }} />;
}
