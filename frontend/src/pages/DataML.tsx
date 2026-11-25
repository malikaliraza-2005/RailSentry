import { AnimatePresence, motion } from "framer-motion";
import { ArrowDownUp, ChevronLeft, ChevronRight, Database, Play, Search, Sparkles, Wrench } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell as BarCell, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { Heatmap, useChartTheme } from "../components/charts";
import {
  Badge, Button, Card, EmptyState, ErrorBanner, KpiCard, Page, ProgressBar, RiskBadge, Skeleton, Tabs,
} from "../components/ui";
import { api, RISK_NAMES } from "../lib/api";
import { useStore } from "../lib/store";
import { useColors } from "../lib/theme";
import { useJob } from "../lib/useJob";

/* eslint-disable @typescript-eslint/no-explicit-any */
type Module2 = any;

export default function DataML() {
  const { results, setResult, refreshOverview } = useStore();
  const data: Module2 | undefined = results.module2;
  const [location, setLocation] = useState({ grid_x: 7, grid_y: 7 });
  const job = useJob<Module2>("module2", (result) => {
    setResult("module2", result);
    refreshOverview();
  });

  const runButton = (
    <Button variant="primary" icon={<Play className="h-4 w-4" />} loading={job.running} onClick={() => job.start(location)}>
      {data ? "Run again" : "Run Module 2"}
    </Button>
  );

  return (
    <Page
      title="Data & Machine Learning"
      subtitle="360 past incidents: what was wrong with the records, what they show, and how well similar cases predict new ones."
      actions={runButton}
    >
      <ErrorBanner message={job.error} onClose={job.clearError} />
      {job.running && (
        <Card className="mb-6"><ProgressBar value={job.job?.progress ?? 0.05} label={job.job?.step || "Starting…"} /></Card>
      )}

      {!data && !job.running && (
        <EmptyState icon={<Database className="h-5 w-5" />} title="No Module 2 results yet"
                    text="Run Module 2 to clean the data, draw the charts and train the KNN classifier and regressor. It takes a few seconds."
                    action={runButton} />
      )}
      {!data && job.running && <Skeleton className="h-96" />}

      {data && (
        <div className="space-y-14">
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <KpiCard label="Clean records" value={data.records_clean} icon={<Database className="h-4 w-4" />}
                     hint={`${data.records_raw} raw records loaded`} />
            <KpiCard label="Values corrected" icon={<Wrench className="h-4 w-4" />} tone="warning" delay={0.04}
                     value={Object.values(data.cleaning_report).reduce((n: number, r: any) => n + r.values.length, 0)}
                     hint="Impossible or extreme values set to missing" />
            <KpiCard label="KNN classifier F1" value={data.knn_classifier.test_macro_f1} decimals={3} tone="success" delay={0.08}
                     icon={<Sparkles className="h-4 w-4" />}
                     hint={`k=${data.knn_classifier.best_k}, ${data.knn_classifier.best_metric} · accuracy ${(data.knn_classifier.test_accuracy * 100).toFixed(1)}%`} />
            <KpiCard label="Response-time MAE" value={data.knn_regressor.mae} decimals={2} suffix=" min" tone="danger" delay={0.12}
                     icon={<Sparkles className="h-4 w-4" />}
                     hint={`k=${data.knn_regressor.best_k}, ${data.knn_regressor.best_metric} · RMSE ${data.knn_regressor.rmse.toFixed(2)} min`} />
          </div>

          <CleaningReport report={data.cleaning_report} />
          <ChartsCard data={data} />
          <div className="grid gap-x-10 gap-y-12 xl:grid-cols-2">
            <ClusterMap data={data} location={location} setLocation={setLocation} />
            <KnnTuning data={data} />
          </div>
          <DatasetTable />
        </div>
      )}
    </Page>
  );
}

function CleaningReport({ report }: { report: Record<string, { reason: string; values: number[]; incident_ids: string[] }> }) {
  const entries = Object.entries(report);
  return (
    <Card title="Data cleaning report" subtitle="Values outside their valid range are data errors. They are set to missing and filled by the model's imputer, using training data only.">
      {entries.length === 0 ? <p className="text-sm text-muted">No problems found.</p> : (
        <div className="grid gap-3 md:grid-cols-2">
          {entries.map(([column, info], index) => (
            <motion.div key={column} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.05 }}
                        className="rounded-lg border border-warning/30 bg-warning/5 p-4">
              <div className="flex items-center justify-between gap-2">
                <span className="num text-sm font-semibold">{column}</span>
                <Badge tone="warning">{info.values.length} fixed</Badge>
              </div>
              <p className="mt-1 text-sm text-muted">{info.reason}</p>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {info.values.map((value, i) => (
                  <span key={i} className="num rounded bg-surface-2 px-1.5 py-0.5 text-[13px]" title={info.incident_ids[i]}>
                    {info.incident_ids[i]}: {value}
                  </span>
                ))}
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </Card>
  );
}

function ChartsCard({ data }: { data: Module2 }) {
  const chart = useChartTheme();
  const [tab, setTab] = useState<"risk" | "hist" | "corr" | "box" | "elbow">("risk");
  const riskData = RISK_NAMES.map((name) => ({ name, value: data.risk_counts[name] }));
  const riskColors = [chart.colors.success, chart.colors.warning, chart.colors.danger];
  const hist = data.histogram.map((b: any) => ({ range: `${b.from.toFixed(1)}`, count: b.count, label: `${b.from.toFixed(1)}–${b.to.toFixed(1)} min` }));

  return (
    <Card
      title="Exploratory charts"
      action={<Tabs id="m2-chart" size="sm" value={tab} onChange={setTab} options={[
        { value: "risk", label: "Risk" }, { value: "hist", label: "Response time" }, { value: "corr", label: "Correlation" },
        { value: "box", label: "Boxplots" }, { value: "elbow", label: "Elbow" },
      ]} />}
    >
      <AnimatePresence mode="wait">
        <motion.div key={tab} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.18 }}>
          {tab === "risk" && (
            <div className="h-72">
              <ResponsiveContainer>
                <BarChart data={riskData}>
                  <CartesianGrid {...chart.grid} />
                  <XAxis dataKey="name" {...chart.axis} />
                  <YAxis {...chart.axis} />
                  <Tooltip {...chart.tooltip} />
                  <Bar dataKey="value" name="Incidents" radius={0} animationDuration={700}>
                    {riskData.map((entry, i) => <BarCell key={entry.name} fill={riskColors[i]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
          {tab === "hist" && (
            <div className="h-72">
              <ResponsiveContainer>
                <BarChart data={hist} barCategoryGap={2}>
                  <CartesianGrid {...chart.grid} />
                  <XAxis dataKey="range" {...chart.axis} label={{ value: "minutes", position: "insideBottomRight", offset: -4, fill: chart.colors.muted, fontSize: 11 }} />
                  <YAxis {...chart.axis} />
                  <Tooltip {...chart.tooltip} labelFormatter={(_, payload) => payload?.[0]?.payload.label ?? ""} />
                  <Bar dataKey="count" name="Incidents" fill={chart.colors.accent} radius={0} animationDuration={700} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
          {tab === "corr" && (
            <Heatmap rows={data.correlation.columns} columns={data.correlation.columns.map((c: string) => c.split("_")[0])}
                     values={data.correlation.matrix} />
          )}
          {tab === "box" && <Boxplots boxes={data.boxplot} />}
          {tab === "elbow" && (
            <div className="h-72">
              <ResponsiveContainer>
                <LineChart data={data.elbow}>
                  <CartesianGrid {...chart.grid} />
                  <XAxis dataKey="clusters" {...chart.axis} />
                  <YAxis {...chart.axis} />
                  <Tooltip {...chart.tooltip} />
                  <Line type="monotone" dataKey="inertia" name="Inertia" stroke={chart.colors.accent} strokeWidth={2.5} dot={{ r: 4 }} animationDuration={800} />
                </LineChart>
              </ResponsiveContainer>
              <p className="text-sm text-muted">3 clusters are used: inertia falls more slowly after this point.</p>
            </div>
          )}
        </motion.div>
      </AnimatePresence>
    </Card>
  );
}

function Boxplots({ boxes }: { boxes: Record<string, { min: number; q1: number; median: number; q3: number; max: number; outliers: number[] } | null> }) {
  const colors = useColors();
  return (
    <div className="space-y-3">
      <p className="text-sm text-muted">Raw data, before cleaning. Each row uses its own scale; red dots are outliers beyond 1.5 × IQR.</p>
      {Object.entries(boxes).map(([column, box]) => {
        if (!box) return null;
        const low = Math.min(box.min, ...box.outliers);
        const high = Math.max(box.max, ...box.outliers);
        const x = (v: number) => ((v - low) / (high - low || 1)) * 100;
        return (
          <div key={column} className="grid grid-cols-[9rem_1fr_6rem] items-center gap-3 text-sm">
            <span className="num truncate text-muted" title={column}>{column}</span>
            <svg viewBox="0 0 100 10" preserveAspectRatio="none" className="h-6 w-full overflow-visible">
              <line x1={x(box.min)} x2={x(box.max)} y1={5} y2={5} stroke={colors.muted} strokeWidth={0.4} vectorEffect="non-scaling-stroke" />
              <motion.rect initial={{ scaleX: 0 }} animate={{ scaleX: 1 }} transition={{ duration: 0.5 }}
                           style={{ transformBox: "fill-box", transformOrigin: "left" }}
                           x={x(box.q1)} y={1.5} width={Math.max(0.5, x(box.q3) - x(box.q1))} height={7} rx={1}
                           fill={colors.alpha("accent", 0.3)} stroke={colors.accent} strokeWidth={1} vectorEffect="non-scaling-stroke" />
              <line x1={x(box.median)} x2={x(box.median)} y1={1.5} y2={8.5} stroke={colors.text} strokeWidth={2} vectorEffect="non-scaling-stroke" />
              {box.outliers.map((value, i) => (
                <circle key={i} cx={x(value)} cy={5} r={1.2} fill={colors.danger} />
              ))}
            </svg>
            <span className="num text-right text-muted">{box.min.toFixed(0)}–{box.max.toFixed(0)}</span>
          </div>
        );
      })}
    </div>
  );
}

function ClusterMap({ data, location, setLocation }: { data: Module2; location: { grid_x: number; grid_y: number }; setLocation: (l: { grid_x: number; grid_y: number }) => void }) {
  const colors = useColors();
  const [predicted, setPredicted] = useState<{ x: number; y: number; cluster: number; id: number } | null>(
    data.new_location_cluster !== null && data.new_location_cluster !== undefined
      ? { x: location.grid_x, y: location.grid_y, cluster: data.new_location_cluster, id: 0 } : null,
  );
  const [error, setError] = useState<string | null>(null);
  // Not the risk colours: a cluster is only a place, not a danger level.
  const clusterColors = [colors.path, colors.accent, colors.muted, colors.success, colors.warning];
  const S = 30;
  const px = (v: number) => v * S + S / 2;

  const onClick = async (event: React.MouseEvent<SVGSVGElement>) => {
    const svg = event.currentTarget;
    const rect = svg.getBoundingClientRect();
    const gx = Math.round(Math.max(0, Math.min(14, ((event.clientX - rect.left) / rect.width) * 15 - 0.5)));
    const gy = Math.round(Math.max(0, Math.min(14, ((event.clientY - rect.top) / rect.height) * 15 - 0.5)));
    try {
      const result = await api.post<{ cluster: number }>("/api/cluster", { grid_x: gx, grid_y: gy });
      setPredicted({ x: gx, y: gy, cluster: result.cluster, id: Date.now() });
      setLocation({ grid_x: gx, grid_y: gy });
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <Card title="Location clusters (K-Means)" subtitle="Click anywhere on the map to assign a new location to a cluster."
          action={predicted ? <Badge tone="accent">({predicted.x}, {predicted.y}) → cluster {predicted.cluster}</Badge> : undefined}>
      <ErrorBanner message={error} />
      <svg viewBox={`0 0 ${15 * S} ${15 * S}`} className="h-auto w-full cursor-crosshair rounded-lg bg-surface-2" onClick={onClick}
           role="img" aria-label="Station location clusters">
        {Array.from({ length: 16 }).map((_, i) => (
          <g key={i}>
            <line x1={i * S} x2={i * S} y1={0} y2={15 * S} stroke={colors.border} strokeWidth={0.6} />
            <line y1={i * S} y2={i * S} x1={0} x2={15 * S} stroke={colors.border} strokeWidth={0.6} />
          </g>
        ))}
        {data.clusters.points.map((p: any, i: number) => (
          <motion.circle key={i} cx={px(p.grid_x) + ((i * 7) % 9) - 4} cy={px(p.grid_y) + ((i * 5) % 9) - 4} r={4.5}
                         fill={clusterColors[p.location_cluster % clusterColors.length]} fillOpacity={0.75}
                         initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: Math.min(0.6, i * 0.002) }} />
        ))}
        {data.clusters.centroids.map((c: any) => (
          <g key={c.cluster}>
            <circle cx={px(c.grid_x)} cy={px(c.grid_y)} r={13} fill={colors.surface} stroke={clusterColors[c.cluster]} strokeWidth={3} />
            <text x={px(c.grid_x)} y={px(c.grid_y) + 4} textAnchor="middle" fontSize={11} fontWeight={700} fill={colors.text}>{c.cluster}</text>
          </g>
        ))}
        {predicted && (
          <g key={predicted.id}>
            <circle cx={px(predicted.x)} cy={px(predicted.y)} r={10} fill="none" stroke={clusterColors[predicted.cluster]} strokeWidth={3} className="ping-ring" />
            <motion.circle cx={px(predicted.x)} cy={px(predicted.y)} r={8} fill={clusterColors[predicted.cluster]} stroke={colors.bg} strokeWidth={2.5}
                           initial={{ scale: 0 }} animate={{ scale: 1 }} style={{ transformBox: "fill-box", transformOrigin: "center" }}
                           transition={{ type: "spring", stiffness: 400, damping: 15 }} />
          </g>
        )}
      </svg>
      <div className="mt-3 flex flex-wrap gap-4 text-sm text-muted">
        {data.clusters.centroids.map((c: any) => (
          <span key={c.cluster} className="inline-flex items-center gap-1.5">
            <span className="h-3 w-3" style={{ background: clusterColors[c.cluster] }} />
            Cluster {c.cluster} · centre ({c.grid_x.toFixed(1)}, {c.grid_y.toFixed(1)})
          </span>
        ))}
      </div>
    </Card>
  );
}

function KnnTuning({ data }: { data: Module2 }) {
  const chart = useChartTheme();
  const tuning = data.knn_classifier.tuning as { k: number; metric: string; accuracy: number; macro_f1: number }[];
  const ks = [...new Set(tuning.map((t) => t.k))];
  const rows = ks.map((k) => {
    const row: Record<string, number> = { k };
    for (const t of tuning.filter((x) => x.k === k)) {
      row[`${t.metric} F1`] = t.macro_f1;
      row[`${t.metric} accuracy`] = t.accuracy;
    }
    return row;
  });
  return (
    <Card title="KNN tuning (validation set)" subtitle={`Chosen by macro F1: k=${data.knn_classifier.best_k}, ${data.knn_classifier.best_metric}. Macro F1 treats the rare High class fairly.`}>
      <div className="h-80">
        <ResponsiveContainer>
          <LineChart data={rows}>
            <CartesianGrid {...chart.grid} />
            <XAxis dataKey="k" {...chart.axis} />
            <YAxis domain={[0.3, 1]} {...chart.axis} />
            <Tooltip {...chart.tooltip} formatter={(v: number) => v.toFixed(3)} />
            <Legend {...chart.legend} />
            <Line dataKey="euclidean F1" stroke={chart.colors.accent} strokeWidth={2.5} dot={{ r: 3 }} animationDuration={800} />
            <Line dataKey="manhattan F1" stroke={chart.colors.path} strokeWidth={2.5} dot={{ r: 3 }} animationDuration={800} />
            <Line dataKey="euclidean accuracy" stroke={chart.colors.accent} strokeDasharray="4 4" strokeWidth={1.5} dot={false} animationDuration={800} />
            <Line dataKey="manhattan accuracy" stroke={chart.colors.path} strokeDasharray="4 4" strokeWidth={1.5} dot={false} animationDuration={800} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

const COLUMNS = [
  "incident_id", "grid_x", "grid_y", "incident_type", "crowd_density_pct", "hazard_severity_1_5",
  "equipment_health_pct", "obstruction_level_0_3", "hour_of_day", "risk_level", "response_time_min",
];

function DatasetTable() {
  const [records, setRecords] = useState<Record<string, any>[] | null>(null);
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<{ column: string; dir: 1 | -1 }>({ column: "incident_id", dir: 1 });
  const [page, setPage] = useState(0);
  const [onlyCleaned, setOnlyCleaned] = useState(false);
  const pageSize = 12;

  useEffect(() => {
    api.get<{ records: Record<string, any>[] }>("/api/dataset").then((d) => setRecords(d.records)).catch(() => setRecords([]));
  }, []);

  const filtered = useMemo(() => {
    if (!records) return [];
    const q = query.trim().toLowerCase();
    return records
      .filter((r) => (!onlyCleaned || r.cleaned_fields.length || r.missing_fields.length))
      .filter((r) => !q || COLUMNS.some((c) => String(r[c] ?? "").toLowerCase().includes(q)))
      .sort((a, b) => {
        const x = a[sort.column], y = b[sort.column];
        if (x === null || x === undefined) return 1;
        if (y === null || y === undefined) return -1;
        return (x > y ? 1 : x < y ? -1 : 0) * sort.dir;
      });
  }, [records, query, sort, onlyCleaned]);

  useEffect(() => setPage(0), [query, sort, onlyCleaned]);
  const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const rows = filtered.slice(page * pageSize, page * pageSize + pageSize);

  return (
    <Card title="Raw dataset" subtitle="Amber cells were corrected during cleaning; empty cells are missing values (filled later by the imputer).">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <label className="relative flex-1 sm:max-w-xs">
          <Search className="pointer-events-none absolute left-2.5 top-2.5 h-4 w-4 text-muted" />
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search records…" aria-label="Search records"
                 className="focus-ring h-9 w-full rounded-lg border border-border bg-surface-2 pl-8 pr-3 text-sm placeholder:text-muted" />
        </label>
        <Button size="sm" variant={onlyCleaned ? "primary" : "secondary"} onClick={() => setOnlyCleaned((v) => !v)}>
          {onlyCleaned ? "Showing problem rows" : "Show only problem rows"}
        </Button>
        <span className="num ml-auto text-sm text-muted">{filtered.length} records</span>
      </div>
      {!records ? <Skeleton className="h-80" /> : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[900px] text-sm">
            <thead>
              <tr className="border-b border-border text-left text-muted">
                {COLUMNS.map((column) => (
                  <th key={column} className="whitespace-nowrap py-2 pr-3 font-medium">
                    <button className="focus-ring inline-flex items-center gap-1 rounded hover:text-text"
                            onClick={() => setSort((s) => ({ column, dir: s.column === column ? (-s.dir as 1 | -1) : 1 }))}>
                      {column.replace(/_/g, " ")}
                      <ArrowDownUp className={`h-3 w-3 ${sort.column === column ? "text-accent" : "opacity-40"}`} />
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((record) => (
                <tr key={record.incident_id} className="border-b border-border/50 hover:bg-surface-2/60">
                  {COLUMNS.map((column) => {
                    const cleaned = record.cleaned_fields.includes(column);
                    const missing = record[column] === null || record[column] === undefined;
                    return (
                      <td key={column} className={`num whitespace-nowrap py-1.5 pr-3 ${cleaned ? "font-semibold text-warning" : ""}`}
                          title={cleaned ? "Corrected during cleaning" : missing ? "Missing value" : undefined}>
                        {column === "risk_level" ? <RiskBadge risk={record[column]} size="sm" />
                          : missing ? <span className="text-muted">—</span>
                          : String(record[column])}
                        {cleaned && <Wrench className="ml-1 inline h-3 w-3" />}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="mt-3 flex items-center justify-end gap-2">
        <Button size="sm" variant="ghost" disabled={page === 0} onClick={() => setPage((p) => p - 1)} icon={<ChevronLeft className="h-4 w-4" />}>Prev</Button>
        <span className="num text-sm text-muted">{page + 1} / {pages}</span>
        <Button size="sm" variant="ghost" disabled={page >= pages - 1} onClick={() => setPage((p) => p + 1)}>Next <ChevronRight className="h-4 w-4" /></Button>
      </div>
    </Card>
  );
}
