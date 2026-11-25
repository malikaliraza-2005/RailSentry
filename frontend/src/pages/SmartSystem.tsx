import { AnimatePresence, LayoutGroup, motion } from "framer-motion";
import { BellRing, BrainCircuit, FlaskConical, ListOrdered, Play, RefreshCw, Route, ShieldCheck } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Link } from "react-router-dom";
import { useChartTheme } from "../components/charts";
import { PlaybackControls } from "../components/PlaybackControls";
import { GridLegend, StationGrid } from "../components/StationGrid";
import {
  Badge, Button, Card, EmptyState, ErrorBanner, Page, ProgressBar, RiskBadge, Skeleton, Stat, Tabs, fmt,
} from "../components/ui";
import { api, ApiError, RISK_NAMES, type Cell, type GridData } from "../lib/api";
import { useStore } from "../lib/store";
import { useColors } from "../lib/theme";
import { useJob } from "../lib/useJob";
import { usePlayback } from "../lib/usePlayback";

/* eslint-disable @typescript-eslint/no-explicit-any */

interface Routing {
  grid: GridData;
  start: Cell;
  goal: Cell;
  actual_map: number[][];
  predicted_map: number[][];
  normal_path: Cell[];
  risk_aware_path: Cell[];
  table: Record<string, any>[];
}

interface Replan {
  grid: GridData;
  start: Cell;
  goal: Cell;
  route_before: Cell[];
  route_after: Cell[] | null;
  current_index: number;
  new_block: Cell;
  cost_before: number;
  cost_after: number | null;
}

export default function SmartSystem() {
  const [routing, setRouting] = useState<Routing | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [missingModels, setMissingModels] = useState(false);

  useEffect(() => {
    api.get<Routing>("/api/module4/routing")
      .then(setRouting)
      .catch((err) => {
        if (err instanceof ApiError && err.status === 404) setMissingModels(true);
        else setError(err.message);
      });
  }, []);

  return (
    <Page title="Integrated Smart System"
          subtitle="Risk predictions change the route, and the vehicle re-plans when the way is blocked.">
      <ErrorBanner message={error} onClose={() => setError(null)} />
      {missingModels ? (
        <EmptyState icon={<BrainCircuit className="h-5 w-5" />} title="Module 3 models are missing"
                    text="The integrated system needs the saved ANN and KNN regressor. Train them on the Neural Networks page first."
                    action={<Link to="/neural"><Button variant="primary">Go to Neural Networks</Button></Link>} />
      ) : (
        <div className="space-y-14">
          {routing ? <RoutingCard routing={routing} /> : <Skeleton className="h-[520px]" />}
          <div className="grid gap-x-10 gap-y-12 xl:grid-cols-2">
            <ReplanCard />
            <TargetsCard grid={routing?.grid} start={routing?.start} />
          </div>
          <ExperimentsCard />
        </div>
      )}
    </Page>
  );
}

function RoutingCard({ routing }: { routing: Routing }) {
  const colors = useColors();
  const [mapView, setMapView] = useState<"actual" | "predicted">("predicted");
  const [route, setRoute] = useState<"risk" | "normal">("risk");
  const path = route === "risk" ? routing.risk_aware_path : routing.normal_path;
  const playback = usePlayback(path.length - 1, 140);

  useEffect(() => {
    playback.reset();
    const id = window.setTimeout(playback.play, 400);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [route]);

  const rows = routing.table;
  return (
    <Card title="Normal vs risk-aware routing" subtitle="60 real incidents placed on the Normal layout. The agent plans with predicted risk; routes are scored against the true risk."
          action={<div className="flex flex-wrap gap-x-10 gap-y-2">
            <Tabs id="map" size="sm" value={mapView} onChange={setMapView} options={[
              { value: "predicted", label: "Predicted risk" }, { value: "actual", label: "True risk" }]} />
            <Tabs id="route" size="sm" value={route} onChange={setRoute} options={[
              { value: "risk", label: "Drive risk-aware" }, { value: "normal", label: "Drive normal" }]} />
          </div>}>
      <div className="grid gap-x-10 gap-y-12 lg:grid-cols-[minmax(0,1.5fr)_1fr]">
        <div>
          <StationGrid
            grid={routing.grid} start={routing.start} goal={routing.goal}
            hazards={mapView === "actual" ? routing.actual_map : routing.predicted_map}
            paths={[
              { id: "normal", cells: routing.normal_path, color: colors.accent, dashed: true, width: 4 },
              { id: "risk", cells: routing.risk_aware_path, color: colors.path, width: 5, duration: 1.4 },
            ]}
            vehicle={path[Math.min(playback.step, path.length - 1)]}
            label="Normal and risk-aware routes over the hazard map"
          />
          <div className="mt-3"><PlaybackControls playback={playback} total={path.length - 1} label="Cell" /></div>
          <div className="mt-3">
            <GridLegend items={[
              { color: colors.accent, label: "Normal shortest route", shape: "dashed" },
              { color: colors.path, label: "Risk-aware route", shape: "line" },
              { color: colors.warning, label: `${mapView === "actual" ? "True" : "Predicted"} Medium risk`, shape: "triangle" as const },
              { color: colors.danger, label: `${mapView === "actual" ? "True" : "Predicted"} High risk` },
            ]} />
          </div>
        </div>
        <div className="space-y-3">
          {rows.map((row, i) => {
            const risky = row["Risk encountered"] > 0;
            const isRisk = String(row.Route).startsWith("Risk");
            return (
              <motion.div key={row.Route} initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 + i * 0.08 }}
                          className={`rounded-card border p-4 ${isRisk ? "border-path/50 bg-path/5" : "border-border bg-surface-2/50"}`}>
                <div className="mb-3 flex items-center justify-between">
                  <span className="text-sm font-semibold">{row.Route}</span>
                  {risky ? <Badge tone="danger">{row["Risk encountered"]} risk</Badge> : <Badge tone="success"><ShieldCheck className="h-3 w-3" /> No risk</Badge>}
                </div>
                <div className="grid grid-cols-3 gap-2">
                  <Stat label="Length" value={row.Length} />
                  <Stat label="Travel cost" value={fmt(row["Travel cost"], 0)} />
                  <Stat label="Actual time" value={`${fmt(row["Actual response time"], 0)} min`} tone={isRisk ? "success" : undefined} />
                </div>
              </motion.div>
            );
          })}
          <p className="text-sm text-muted">
            Actual response time adds the delay of every truly risky cell crossed (Medium +2 min, High +6 min). The risk-aware route can be
            the same length but avoids hazards the ANN predicted.
          </p>
        </div>
      </div>
    </Card>
  );
}

function ReplanCard() {
  const colors = useColors();
  const [step, setStep] = useState(6);
  const [data, setData] = useState<Replan | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback((s: number) => {
    api.get<Replan>(`/api/module4/replan?step=${s}`).then((d) => { setData(d); setError(null); }).catch((err) => setError(err.message));
  }, []);
  useEffect(() => load(step), [load, step]);

  const total = data ? (data.route_after ?? data.route_before).length - 1 : 0;
  const playback = usePlayback(total, 160);
  useEffect(() => {
    playback.reset();
    const id = window.setTimeout(playback.play, 300);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  if (error) return <Card title="Automatic re-planning"><ErrorBanner message={error} /></Card>;
  if (!data) return <Skeleton className="h-[520px]" />;

  const reached = playback.step >= data.current_index;
  const route = data.route_after ?? data.route_before;
  const position = route[Math.min(playback.step, route.length - 1)];
  const gridView = reached ? data.grid : data.grid.map((row, r) => row.map((v, c) => (r === data.new_block[0] && c === data.new_block[1] ? 1 : v)));

  return (
    <Card title="Automatic re-planning" subtitle="A new obstruction is detected on the cell ahead. The agent re-plans from where it is, not from the start."
          action={reached ? <Badge tone="warning"><BellRing className="h-3 w-3" /> Obstruction detected</Badge> : <Badge>Driving planned route</Badge>}>
      <StationGrid
        grid={gridView} start={data.start} goal={data.goal} vehicle={position}
        paths={[
          { id: "before", cells: data.route_before, color: colors.alpha("accent", reached ? 0.35 : 0.9), dashed: true, width: 4 },
          ...(reached && data.route_after ? [{ id: "after", cells: data.route_after.slice(data.current_index), color: colors.path, width: 5, duration: 0.9 }] : []),
        ]}
        markers={reached ? [{ cell: data.new_block, color: colors.danger, kind: "x" as const, pulse: true, title: "New obstruction" }] : []}
        label="Automatic re-planning demonstration"
      />
      <div className="mt-3"><PlaybackControls playback={playback} total={total} label="Cell" /></div>
      <label className="mt-4 block">
        <div className="mb-1 flex justify-between text-sm"><span className="text-muted">Obstruction appears after step</span><span className="num">{data.current_index}</span></div>
        <input type="range" min={1} max={data.route_before.length - 3} value={step} onChange={(e) => setStep(+e.target.value)}
               className="w-full accent-[rgb(var(--accent))]" aria-label="Step at which the obstruction appears" />
      </label>
      <div className="mt-3 grid grid-cols-3 gap-2">
        <Stat label="Planned cost" value={fmt(data.cost_before, 0)} />
        <Stat label="Cost after re-plan" value={fmt(data.cost_after, 0)} tone="warning" />
        <Stat label="Re-plans" value={reached ? 1 : 0} />
      </div>
      <AnimatePresence>
        {reached && (
          <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                      className="mt-3 flex items-center gap-2 rounded-lg bg-warning/10 px-3 py-2 text-sm text-warning">
            <RefreshCw className="h-4 w-4" /> Blocked cell ({data.new_block[0]}, {data.new_block[1]}) added to the map; new route planned from ({route[data.current_index][0]}, {route[data.current_index][1]}).
          </motion.div>
        )}
      </AnimatePresence>
    </Card>
  );
}

function TargetsCard({ grid, start }: { grid?: GridData; start?: Cell }) {
  const colors = useColors();
  const [targets, setTargets] = useState<any[] | null>(null);
  const [ranked, setRanked] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.get<any[]>("/api/module4/targets").then((t) => {
      setTargets(t);
      window.setTimeout(() => setRanked(true), 900);
    }).catch((err) => setError(err.message));
  }, []);

  // Before ranking, show them in the order they were picked (by incident id).
  const shown = useMemo(() => {
    if (!targets) return [];
    return ranked ? targets : [...targets].sort((a, b) => String(a.name).localeCompare(String(b.name)));
  }, [targets, ranked]);
  const riskColor = [colors.success, colors.warning, colors.danger];

  return (
    <Card title="Multi-target planning" subtitle="Score = 20 × predicted risk − 0.5 × predicted response time − 0.5 × route cost. Highest score is served first."
          action={<Button size="sm" variant="ghost" icon={<ListOrdered className="h-3.5 w-3.5" />} onClick={() => { setRanked(false); window.setTimeout(() => setRanked(true), 500); }}>Re-rank</Button>}>
      <ErrorBanner message={error} />
      {!targets ? <Skeleton className="h-80" /> : (
        <>
          <LayoutGroup>
            <ol className="space-y-2">
              {shown.map((target, index) => (
                <motion.li key={target.name} layout transition={{ type: "spring", stiffness: 380, damping: 30 }}
                           className={`flex items-center gap-3 rounded-lg border px-3 py-2.5 ${ranked && index === 0 ? "border-accent/60 bg-accent/10" : "border-border bg-surface-2/50"}`}>
                  <span className={`num grid h-7 w-7 shrink-0 place-items-center text-sm font-semibold ${ranked ? "bg-accent text-onaccent" : "bg-surface-2 text-muted"}`}>
                    {ranked ? index + 1 : "?"}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 text-sm font-medium">
                      {target.name}
                      <span className="truncate text-sm font-normal text-muted">{String(target.type).replace("_", " ")} · cell ({target.cell[0]}, {target.cell[1]})</span>
                    </div>
                    <div className="num mt-0.5 text-[13px] text-muted">resp. {target.response_time.toFixed(1)} min · route {target.route_cost}</div>
                  </div>
                  <RiskBadge risk={RISK_NAMES[target.predicted_risk]} size="sm" />
                  <span className="num w-14 text-right text-sm font-semibold">{target.score.toFixed(1)}</span>
                </motion.li>
              ))}
            </ol>
          </LayoutGroup>
          {grid && start && (
            <div className="mx-auto mt-4 max-w-[340px]">
              <StationGrid grid={grid} start={start}
                           markers={targets.map((t) => ({ cell: t.cell as Cell, color: riskColor[t.predicted_risk], kind: "ring" as const, title: t.name }))}
                           label="Incident targets on the station map" />
            </div>
          )}
        </>
      )}
    </Card>
  );
}

function ExperimentsCard() {
  const chart = useChartTheme();
  const { overview, results, setResult, refreshOverview } = useStore();
  const [view, setView] = useState<"summary" | "runs">("summary");
  const [metric, setMetric] = useState<"Performance" | "Risk encountered" | "Response time">("Performance");
  const job = useJob<any>("module4", (result) => {
    setResult("module4", result);
    refreshOverview();
  });
  const summary: any[] | undefined = results.module4?.experiments?.summary ?? overview?.results.module4?.summary;
  const runs: any[] | undefined = results.module4?.experiments?.runs;

  const chartRows = useMemo(() => {
    if (!summary) return [];
    const layouts = [...new Set(summary.map((r) => r["Layout name"]))];
    return layouts.map((name) => {
      const row: Record<string, any> = { layout: String(name).replace(" Layout", "") };
      for (const r of summary.filter((x) => x["Layout name"] === name)) row[r.Mode] = r[metric] ?? 0;
      return row;
    });
  }, [summary, metric]);

  return (
    <Card title="Experiments: 10 runs × 3 layouts" subtitle="Each run samples 60 incidents and adds one new obstruction on the route. Both modes face the same events."
          action={<Button variant="primary" size="sm" icon={<FlaskConical className="h-3.5 w-3.5" />} loading={job.running} onClick={() => job.start()}>
            {summary ? "Run again" : "Run experiments"}</Button>}>
      <ErrorBanner message={job.error} onClose={job.clearError} />
      {job.running && <div className="mb-4"><ProgressBar value={job.job?.progress ?? 0.1} label={job.job?.step || "Starting…"} /></div>}
      {!summary ? (
        job.running ? <Skeleton className="h-72" /> : (
          <EmptyState icon={<Route className="h-5 w-5" />} title="No experiment results yet" text="Run the 30 experiments to compare routing with and without ANN predictions."
                      action={<Button variant="primary" icon={<Play className="h-4 w-4" />} onClick={() => job.start()}>Run experiments</Button>} />
        )
      ) : (
        <>
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <Tabs id="exp-metric" size="sm" value={metric} onChange={setMetric} options={[
              { value: "Performance", label: "Performance" }, { value: "Risk encountered", label: "Risk" }, { value: "Response time", label: "Response time" }]} />
            <div className="ml-auto">
              <Tabs id="exp-view" size="sm" value={view} onChange={setView} options={[
                { value: "summary", label: "Averages" }, { value: "runs", label: `All runs${runs ? "" : " (run to load)"}` }]} />
            </div>
          </div>
          <div className="h-72">
            <ResponsiveContainer>
              <BarChart data={chartRows}>
                <CartesianGrid {...chart.grid} />
                <XAxis dataKey="layout" {...chart.axis} />
                <YAxis {...chart.axis} />
                <Tooltip {...chart.tooltip} formatter={(v: number) => v.toFixed(2)} />
                <Legend {...chart.legend} />
                <Bar dataKey="Without predictions" fill={chart.colors.muted} radius={0} animationDuration={700} />
                <Bar dataKey="With predictions" fill={chart.colors.path} radius={0} animationDuration={700} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 overflow-x-auto">
            {view === "summary" || !runs ? (
              <ResultTable rows={summary} columns={["Layout name", "Mode", "Successful target", "Travel cost", "Response time", "Risk encountered", "Re-plans", "Performance"]} />
            ) : (
              <div className="max-h-96 overflow-y-auto">
                <ResultTable rows={runs} columns={["Layout name", "Run", "Mode", "Successful target", "Travel cost", "Response time", "Risk encountered", "Re-plans", "Performance"]} />
              </div>
            )}
          </div>
          <p className="mt-3 text-sm text-muted">The unreachable layout fails in both modes, as expected: no route reaches the walled-in goal.</p>
        </>
      )}
    </Card>
  );
}

function ResultTable({ rows, columns }: { rows: any[]; columns: string[] }) {
  return (
    <table className="w-full min-w-[760px] text-sm">
      <thead className="sticky top-0 bg-surface">
        <tr className="border-b border-border text-left text-muted">
          {columns.map((c) => <th key={c} className={`py-2 pr-3 font-medium ${c === "Layout name" || c === "Mode" ? "" : "text-right"}`}>{c}</th>)}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={i} className={`border-b border-border/50 ${row.Mode === "With predictions" ? "bg-path/5" : ""}`}>
            {columns.map((c) => {
              const v = row[c];
              const text = typeof v === "number" ? (Number.isInteger(v) && c !== "Performance" ? String(v) : v.toFixed(2)) : v ?? "n/a";
              return <td key={c} className={`py-1.5 pr-3 ${c === "Layout name" || c === "Mode" ? "" : "num text-right"}`}>{text}</td>;
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
