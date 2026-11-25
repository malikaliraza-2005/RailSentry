import { AnimatePresence, motion } from "framer-motion";
import { BrainCircuit, Clock, Crown, Gauge, Play, Sparkles, TriangleAlert } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Heatmap, useChartTheme } from "../components/charts";
import {
  Badge, Button, Card, EmptyState, ErrorBanner, Page, ProgressBar, RiskBadge, Skeleton, Tabs,
} from "../components/ui";
import { api, RISK_NAMES, type JobEvent, type RiskName } from "../lib/api";
import { useStore } from "../lib/store";
import { useJob } from "../lib/useJob";

/* eslint-disable @typescript-eslint/no-explicit-any */
type Module3 = any;
const MODELS = ["ANN Architecture 1", "ANN Architecture 2"];

export default function NeuralNetworks() {
  const { results, setResult, refreshOverview } = useStore();
  const data: Module3 | undefined = results.module3;
  const job = useJob<Module3>("module3", (result) => {
    setResult("module3", result);
    refreshOverview();
  });

  const trainButton = (
    <Button variant="primary" icon={<Play className="h-4 w-4" />} loading={job.running} onClick={() => job.start()}>
      {job.running ? "Training…" : data ? "Train again" : "Train models"}
    </Button>
  );

  return (
    <Page
      title="Neural Networks"
      subtitle="Two networks learn to judge risk. The better one is tested on incidents it has never seen."
      actions={trainButton}
    >
      <ErrorBanner message={job.error} onClose={job.clearError} />

      {(job.running || (job.events.length > 0 && job.job?.status !== "done")) && <LiveTraining events={job.events} progress={job.job?.progress ?? 0} step={job.job?.step ?? ""} />}

      <div className="grid gap-x-10 gap-y-12 xl:grid-cols-[1fr_380px]">
        <div className="min-w-0 space-y-14">
          {!data && !job.running && (
            <EmptyState icon={<BrainCircuit className="h-5 w-5" />} title="Models not trained in this session"
                        text="Train the models to see training curves, the comparison table, ROC curves and confusion matrices. Training takes about a minute."
                        action={trainButton} />
          )}
          {!data && job.running && <Skeleton className="h-72" />}
          {data && <Results data={data} />}
        </div>
        <PredictionForm />
      </div>
    </Page>
  );
}

function historyRows(histories: Record<string, Record<string, number[]>>, metric: "accuracy" | "loss") {
  const length = Math.max(0, ...Object.values(histories).map((h) => h[metric]?.length ?? 0));
  return Array.from({ length }, (_, i) => {
    const row: Record<string, number> = { epoch: i + 1 };
    for (const [name, history] of Object.entries(histories)) {
      if (history[metric]?.[i] !== undefined) row[`${name} train`] = history[metric][i];
      if (history[`val_${metric}`]?.[i] !== undefined) row[`${name} validation`] = history[`val_${metric}`][i];
    }
    return row;
  });
}

function eventsToHistories(events: JobEvent[]) {
  const histories: Record<string, Record<string, number[]>> = {};
  for (const event of events) {
    if (!event.model || !event.epoch) continue;
    const h = (histories[event.model] ??= { accuracy: [], val_accuracy: [], loss: [], val_loss: [] });
    h.accuracy[event.epoch - 1] = event.accuracy ?? NaN;
    h.val_accuracy[event.epoch - 1] = event.val_accuracy ?? NaN;
    h.loss[event.epoch - 1] = event.loss ?? NaN;
    h.val_loss[event.epoch - 1] = event.val_loss ?? NaN;
  }
  return histories;
}

function CurveChart({ histories, metric, height = 260, animate = true }: {
  histories: Record<string, Record<string, number[]>>; metric: "accuracy" | "loss"; height?: number; animate?: boolean;
}) {
  const chart = useChartTheme();
  const rows = historyRows(histories, metric);
  const colorsFor = [chart.colors.accent, chart.colors.path];
  return (
    <div style={{ height }}>
      <ResponsiveContainer>
        <LineChart data={rows}>
          <CartesianGrid {...chart.grid} />
          <XAxis dataKey="epoch" {...chart.axis} />
          <YAxis {...chart.axis} domain={metric === "accuracy" ? [0, 1] : [0, "auto"]} />
          <Tooltip {...chart.tooltip} formatter={(v: number) => v.toFixed(4)} />
          <Legend {...chart.legend} />
          {MODELS.filter((m) => histories[m]).flatMap((model, i) => [
            <Line key={`${model}t`} dataKey={`${model} train`} name={`${model.replace("ANN Architecture", "ANN")} train`}
                  stroke={colorsFor[i]} strokeWidth={2.2} dot={false} isAnimationActive={animate} animationDuration={600} />,
            <Line key={`${model}v`} dataKey={`${model} validation`} name={`${model.replace("ANN Architecture", "ANN")} validation`}
                  stroke={colorsFor[i]} strokeDasharray="5 4" strokeWidth={1.8} dot={false} isAnimationActive={animate} animationDuration={600} />,
          ])}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function LiveTraining({ events, progress, step }: { events: JobEvent[]; progress: number; step: string }) {
  const histories = useMemo(() => eventsToHistories(events), [events]);
  const last = [...events].reverse().find((e) => e.epoch);
  return (
    <Card title="Live training" subtitle="Curves update after every epoch. Early stopping watches validation loss (patience 8)." className="mb-6"
          action={last ? <Badge tone="accent"><Sparkles className="h-3 w-3" /> {last.model} · epoch {last.epoch}</Badge> : undefined}>
      <ProgressBar value={progress} label={step || "Preparing data…"} />
      <div className="mt-5 grid gap-x-10 gap-y-12 lg:grid-cols-2">
        <div>
          <div className="mb-2 text-sm font-medium text-muted">Accuracy</div>
          <CurveChart histories={histories} metric="accuracy" animate={false} height={220} />
        </div>
        <div>
          <div className="mb-2 text-sm font-medium text-muted">Loss</div>
          <CurveChart histories={histories} metric="loss" animate={false} height={220} />
        </div>
      </div>
    </Card>
  );
}

function Results({ data }: { data: Module3 }) {
  const chart = useChartTheme();
  const [curve, setCurve] = useState<"accuracy" | "loss">("accuracy");
  const [rocClass, setRocClass] = useState<RiskName>("High");
  const comparison = data.comparison as Record<string, any>[];
  const metrics = ["accuracy", "precision", "recall", "macro_F1", "ROC-AUC"];
  const bestPerColumn = Object.fromEntries(metrics.map((m) => [m, Math.max(...comparison.map((r) => r[m]))]));
  const rocModels = Object.keys(data.roc);
  const rocColors = [chart.colors.warning, chart.colors.path];

  const rocRows = useMemo(() => {
    const rows: Record<string, number>[] = [];
    rocModels.forEach((model) => {
      for (const point of data.roc[model][rocClass]) rows.push({ fpr: point.fpr, [model]: point.tpr });
    });
    rows.push({ fpr: 0, chance: 0 }, { fpr: 1, chance: 1 });
    return rows.sort((a, b) => a.fpr - b.fpr);
  }, [data, rocClass, rocModels]);

  return (
    <>
      <Card title="Model comparison (test set)" subtitle={`Train ${data.split_sizes.train} · validation ${data.split_sizes.validation} · test ${data.split_sizes.test} incidents. Best value in each column is highlighted.`}
            action={<Badge tone="success"><Crown className="h-3 w-3" /> {data.best_ann}</Badge>}>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[560px] text-sm">
            <thead>
              <tr className="border-b border-border text-left text-sm text-muted">
                <th className="py-2 font-medium">Model</th>
                {metrics.map((m) => <th key={m} className="py-2 text-right font-medium">{m.replace("_", " ")}</th>)}
              </tr>
            </thead>
            <tbody>
              {comparison.map((row, i) => (
                <motion.tr key={row.Model} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.06 }}
                           className="border-b border-border/60">
                  <td className="py-2.5 font-medium">
                    {row.Model} {row.Model === data.best_ann && <Crown className="ml-1 inline h-3.5 w-3.5 text-warning" />}
                  </td>
                  {metrics.map((m) => (
                    <td key={m} className="num py-2.5 text-right">
                      <span className={row[m] === bestPerColumn[m] ? "rounded bg-success/15 px-1.5 py-0.5 font-semibold text-success" : ""}>
                        {row[m].toFixed(3)}
                      </span>
                    </td>
                  ))}
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-4 grid gap-2 sm:grid-cols-3">
          {MODELS.map((model) => data.validation_metrics[model] && (
            <div key={model} className="rounded-lg bg-surface-2 px-3 py-2 text-sm">
              <div className="text-muted">{model} · validation</div>
              <div className="num mt-0.5 font-semibold">F1 {data.validation_metrics[model].macro_f1.toFixed(3)} · acc {data.validation_metrics[model].accuracy.toFixed(3)}</div>
            </div>
          ))}
          <div className="rounded-lg bg-surface-2 px-3 py-2 text-sm">
            <div className="text-muted">KNN (k={data.knn.k}, {data.knn.metric}) regressor</div>
            <div className="num mt-0.5 font-semibold">MAE {data.regressor.mae.toFixed(2)} min</div>
          </div>
        </div>
      </Card>

      <Card title="Training curves" action={<Tabs id="curve" size="sm" value={curve} onChange={setCurve}
            options={[{ value: "accuracy", label: "Accuracy" }, { value: "loss", label: "Loss" }]} />}>
        <CurveChart histories={data.histories} metric={curve} />
      </Card>

      <div className="grid gap-x-10 gap-y-12 2xl:grid-cols-2">
        <Card title="ROC curves (one-vs-rest)" action={<Tabs id="roc" size="sm" value={rocClass} onChange={setRocClass}
              options={RISK_NAMES.map((r) => ({ value: r, label: r }))} />}>
          <div className="h-72">
            <ResponsiveContainer>
              <LineChart data={rocRows}>
                <CartesianGrid {...chart.grid} vertical />
                <XAxis type="number" dataKey="fpr" domain={[0, 1]} {...chart.axis} label={{ value: "False positive rate", position: "insideBottom", offset: -2, fill: chart.colors.muted, fontSize: 11 }} />
                <YAxis domain={[0, 1]} {...chart.axis} />
                <Tooltip {...chart.tooltip} formatter={(v: number) => v.toFixed(3)} labelFormatter={(v) => `FPR ${Number(v).toFixed(3)}`} />
                <Legend {...chart.legend} />
                {rocModels.map((model, i) => (
                  <Line key={model} dataKey={model} type="stepAfter" stroke={rocColors[i % 2]} strokeWidth={2.4} dot={false} connectNulls animationDuration={700} />
                ))}
                <Line dataKey="chance" name="Chance" stroke={chart.colors.muted} strokeDasharray="3 4" dot={false} connectNulls isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Confusion matrices" subtitle="Rows: actual risk · columns: predicted risk">
          <div className="space-y-5">
            {Object.entries(data.confusion_matrices).map(([model, matrix]) => {
              const values = matrix as number[][];
              return (
                <div key={model}>
                  <div className="mb-1.5 text-sm font-medium">{model}</div>
                  <Heatmap rows={RISK_NAMES} columns={RISK_NAMES} values={values} min={0} max={Math.max(...values.flat())}
                           diverging={false} format={(v) => String(v)} rowTitle="actual ↓" />
                </div>
              );
            })}
          </div>
        </Card>
      </div>

      <Card title="Wrong predictions (best ANN, test set)" subtitle={`${data.wrong_predictions.length} mistakes. Under-estimating risk is the dangerous direction for routing.`}>
        {data.wrong_predictions.length === 0 ? <p className="text-sm text-muted">No wrong predictions.</p> : (
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {data.wrong_predictions.slice(0, 6).map((w: any, i: number) => (
              <motion.div key={i} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}
                          className={`rounded-lg border p-3 ${w.under_estimated ? "border-danger/40 bg-danger/5" : "border-warning/40 bg-warning/5"}`}>
                <div className="flex items-center gap-2 text-sm">
                  <span className="text-muted">Actual</span><RiskBadge risk={w.actual} size="sm" />
                  <span className="text-muted">→</span><RiskBadge risk={w.predicted} size="sm" />
                </div>
                <div className="mt-2 text-sm font-medium">{String(w.incident_type).replace("_", " ")}</div>
                <div className="num mt-1 text-[13px] text-muted">
                  severity {w.hazard_severity_1_5} · crowd {w.crowd_density_pct ?? "—"}% · obstruction {w.obstruction_level_0_3 ?? "—"}
                </div>
                <p className={`mt-2 flex gap-1.5 text-sm ${w.under_estimated ? "text-danger" : "text-warning"}`}>
                  <TriangleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  {w.under_estimated ? "Risk under-estimated: vehicle may be routed through danger." : "Risk over-estimated: vehicle may detour and arrive later."}
                </p>
              </motion.div>
            ))}
          </div>
        )}
      </Card>
    </>
  );
}

const TYPES = ["Structural", "Medical", "Crowd_Control", "Electrical", "Equipment_Failure", "Fire_Smoke"];

function Field({ label, value, children }: { label: string; value?: ReactNode; children: ReactNode }) {
  return (
    <label className="block">
      <div className="mb-1 flex justify-between text-sm"><span className="text-muted">{label}</span><span className="num font-medium">{value}</span></div>
      {children}
    </label>
  );
}

function PredictionForm() {
  const [form, setForm] = useState({
    grid_x: 5, grid_y: 6, incident_type: "Fire_Smoke", crowd_density_pct: 85, hazard_severity_1_5: 4,
    equipment_health_pct: 40, obstruction_level_0_3: 2, hour_of_day: 18,
  });
  const [result, setResult] = useState<{ predicted_risk: RiskName; probabilities: Record<RiskName, number>; response_time_min: number | null } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const set = (key: keyof typeof form, value: number | string) => setForm((f) => ({ ...f, [key]: value }));
  const slider = "w-full accent-[rgb(var(--accent))]";

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      setResult(await api.post("/api/predict", form));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="xl:sticky xl:top-28 xl:self-start">
      <Card title="Predict a new incident" subtitle="Uses the saved best ANN for risk and the KNN regressor for response time.">
        <form onSubmit={submit} className="space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <Field label="Grid X" value={form.grid_x}>
              <input type="range" min={0} max={14} value={form.grid_x} onChange={(e) => set("grid_x", +e.target.value)} className={slider} />
            </Field>
            <Field label="Grid Y" value={form.grid_y}>
              <input type="range" min={0} max={14} value={form.grid_y} onChange={(e) => set("grid_y", +e.target.value)} className={slider} />
            </Field>
          </div>
          <Field label="Incident type">
            <select value={form.incident_type} onChange={(e) => set("incident_type", e.target.value)}
                    className="focus-ring h-9 w-full rounded-lg border border-border bg-surface-2 px-2 text-sm">
              {TYPES.map((t) => <option key={t} value={t}>{t.replace("_", " ")}</option>)}
            </select>
          </Field>
          <Field label="Crowd density" value={`${form.crowd_density_pct}%`}>
            <input type="range" min={0} max={100} value={form.crowd_density_pct} onChange={(e) => set("crowd_density_pct", +e.target.value)} className={slider} />
          </Field>
          <Field label="Hazard severity" value={`${form.hazard_severity_1_5} / 5`}>
            <div className="grid grid-cols-5 gap-1">
              {[1, 2, 3, 4, 5].map((n) => (
                <button type="button" key={n} onClick={() => set("hazard_severity_1_5", n)}
                        className={`focus-ring num h-8 rounded-md text-sm font-medium transition-colors ${
                          form.hazard_severity_1_5 === n ? "bg-accent text-onaccent" : "bg-surface-2 text-muted hover:text-text"}`}>{n}</button>
              ))}
            </div>
          </Field>
          <Field label="Equipment health" value={`${form.equipment_health_pct}%`}>
            <input type="range" min={0} max={100} value={form.equipment_health_pct} onChange={(e) => set("equipment_health_pct", +e.target.value)} className={slider} />
          </Field>
          <Field label="Obstruction level" value={`${form.obstruction_level_0_3} / 3`}>
            <div className="grid grid-cols-4 gap-1">
              {[0, 1, 2, 3].map((n) => (
                <button type="button" key={n} onClick={() => set("obstruction_level_0_3", n)}
                        className={`focus-ring num h-8 rounded-md text-sm font-medium transition-colors ${
                          form.obstruction_level_0_3 === n ? "bg-accent text-onaccent" : "bg-surface-2 text-muted hover:text-text"}`}>{n}</button>
              ))}
            </div>
          </Field>
          <Field label="Hour of day" value={`${String(form.hour_of_day).padStart(2, "0")}:00`}>
            <input type="range" min={0} max={23} value={form.hour_of_day} onChange={(e) => set("hour_of_day", +e.target.value)} className={slider} />
          </Field>
          <ErrorBanner message={error} onClose={() => setError(null)} />
          <Button type="submit" variant="primary" className="w-full" loading={loading} icon={<Gauge className="h-4 w-4" />}>Predict risk</Button>
        </form>

        <AnimatePresence mode="wait">
          {result && (
            <motion.div key={JSON.stringify(result)} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                        className="mt-5 space-y-3 border-t border-border pt-5" aria-live="polite">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted">Predicted risk</span>
                <RiskBadge risk={result.predicted_risk} />
              </div>
              {RISK_NAMES.map((name, i) => {
                const p = result.probabilities[name];
                const color = ["bg-success", "bg-warning", "bg-danger"][i];
                return (
                  <div key={name}>
                    <div className="mb-1 flex justify-between text-sm"><span className="text-muted">{name}</span><span className="num">{(p * 100).toFixed(1)}%</span></div>
                    <div className="h-2 overflow-hidden rounded-none bg-surface-2">
                      <motion.div className={`h-full rounded-none ${color}`} initial={{ width: 0 }} animate={{ width: `${p * 100}%` }}
                                  transition={{ duration: 0.6, delay: i * 0.08, ease: "easeOut" }} />
                    </div>
                  </div>
                );
              })}
              {result.response_time_min !== null && (
                <div className="flex items-center justify-between rounded-lg bg-surface-2 px-3 py-2 text-sm">
                  <span className="inline-flex items-center gap-1.5 text-muted"><Clock className="h-4 w-4" />Predicted response time</span>
                  <span className="num font-semibold">{result.response_time_min.toFixed(1)} min</span>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </Card>
    </div>
  );
}
