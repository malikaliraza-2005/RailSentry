import { motion } from "framer-motion";
import { Download, FileUp, MapPin, OctagonAlert, Plus, Sparkles, Trash2, TriangleAlert, Undo2 } from "lucide-react";
import { useEffect, useRef, useState, type DragEvent, type FormEvent } from "react";
import { PlaybackControls } from "../components/PlaybackControls";
import { GridLegend, StationGrid, type Marker } from "../components/StationGrid";
import { Badge, Button, Card, EmptyState, ErrorBanner, Page, RiskBadge, Skeleton, Stat, Tabs, fmt } from "../components/ui";
import { api, cellKey, RISK_NAMES, type Cell, type GridData, type Layout, type RiskName } from "../lib/api";
import { useStore } from "../lib/store";
import { useColors } from "../lib/theme";
import { usePlayback } from "../lib/usePlayback";

const FIELDS = [
  "incident_id", "incident_type", "grid_x", "grid_y", "crowd_density_pct", "hazard_severity_1_5",
  "equipment_health_pct", "obstruction_level_0_3", "hour_of_day",
] as const;
type Field = (typeof FIELDS)[number];
type Row = Record<Field, string>;

const TYPES = ["Structural", "Medical", "Crowd_Control", "Electrical", "Equipment_Failure", "Fire_Smoke"];
const NUMBERS: { field: Field; label: string; min: number; max: number; step: string; optional?: boolean }[] = [
  { field: "grid_x", label: "Grid X (column)", min: 0, max: 14, step: "1" },
  { field: "grid_y", label: "Grid Y (row)", min: 0, max: 14, step: "1" },
  { field: "crowd_density_pct", label: "Crowd %", min: 0, max: 100, step: "any", optional: true },
  { field: "hazard_severity_1_5", label: "Severity", min: 1, max: 5, step: "1" },
  { field: "equipment_health_pct", label: "Equipment %", min: 0, max: 100, step: "any", optional: true },
  { field: "obstruction_level_0_3", label: "Obstruction", min: 0, max: 3, step: "1", optional: true },
  { field: "hour_of_day", label: "Hour", min: 0, max: 23, step: "1" },
];
const EXAMPLES: Row[] = [
  ["FIRE-1", "Fire_Smoke", "6", "5", "85", "5", "40", "3", "18"],
  ["MED-1", "Medical", "3", "9", "40", "2", "80", "0", "10"],
  ["ELEC-1", "Electrical", "11", "4", "60", "4", "35", "2", "21"],
].map((values) => Object.fromEntries(FIELDS.map((field, i) => [field, values[i]])) as Row);
const BLANK: Row = { ...EXAMPLES[0], incident_id: "" };
const MAX_ROWS = 200;
const TOUR = "__tour__";

interface Problem { row: number; field?: string; message: string }
interface RouteInfo { path: Cell[]; travel_cost: number; predicted_time: number; predicted_risk: number }
interface Incident {
  row: number;
  incident_id: string;
  incident_type: string;
  cell: Cell;
  risk_level: RiskName;
  probabilities: Record<RiskName, number>;
  predicted_response_time: number;
  cluster: number;
  reachable: boolean;
  reason: string | null;
  priority: number | null;
  score: number | null;
  normal: RouteInfo | null;
  risk_aware: RouteInfo | null;
}
interface Analysis {
  errors: Problem[];
  warnings: Problem[];
  incidents: Incident[];
  submitted: number;
  grid: GridData;
  start: Cell;
  plan: { predicted_map: number[][]; tour: Cell[]; tour_cost: number } | null;
}
interface Saved { rows: Row[]; result: Analysis | null; grid: GridData | null; start: Cell; layoutName: string }

const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error));

function download(name: string, rows: (string | number | null | undefined)[][]) {
  const quote = (value: string | number | null | undefined) => {
    let text = value === null || value === undefined ? "" : String(value);
    // Stop spreadsheet programs from running text that looks like a formula.
    if (typeof value === "string" && /^[=+\-@\t\r]/.test(text)) text = `'${text}`;
    return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  const csv = "﻿" + rows.map((row) => row.map(quote).join(",")).join("\r\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const pathCells = (path: Cell[]) => path.map((cell) => `(${cell[0]} ${cell[1]})`).join(" ");

export default function TryData() {
  const { results, setResult } = useStore();
  const saved = results.userData as Saved | undefined;
  const [rows, setRows] = useState<Row[]>(saved?.rows ?? []);
  const [result, setAnalysis] = useState<Analysis | null>(saved?.result ?? null);
  const [problems, setProblems] = useState<Problem[]>(saved?.result?.errors ?? []);
  const [layouts, setLayouts] = useState<Record<string, Layout> | null>(null);
  const [layoutName, setLayoutName] = useState(saved?.layoutName ?? "Normal Layout");
  const [grid, setGrid] = useState<GridData | null>(saved?.grid ?? null);
  const [start, setStart] = useState<Cell>(saved?.start ?? [13, 1]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    api.get<Record<string, Layout>>("/api/layouts").then((data) => {
      if (!alive) return;
      setLayouts(data);
      setGrid((current) => current ?? data["Normal Layout"].grid);
    }).catch((err) => alive && setError(messageOf(err)));
    return () => { alive = false; };
  }, []);

  // Keep the data when the user visits another page.
  useEffect(() => {
    setResult("userData", { rows, result, grid, start, layoutName });
  }, [rows, result, grid, start, layoutName, setResult]);

  /** Old results no longer match once the input changes. */
  const changed = () => {
    setAnalysis(null);
    setProblems([]);
    setError(null);
  };
  const addRows = (added: Row[]) => {
    if (rows.length + added.length > MAX_ROWS) {
      setError(`The limit is ${MAX_ROWS} incidents at a time.`);
      return false;
    }
    changed();
    setRows([...rows, ...added]);
    return true;
  };
  const chooseLayout = (name: string) => {
    if (!layouts) return;
    changed();
    setLayoutName(name);
    setGrid(layouts[name].grid);
    setStart(layouts[name].start);
  };

  const analyse = async () => {
    if (!grid) return;
    setLoading(true);
    setError(null);
    try {
      const data = await api.post<Analysis>("/api/analyze", { rows, grid, start });
      setProblems(data.errors);
      setAnalysis(data.incidents.length ? data : null);
      if (!data.incidents.length) setError("None of the incidents are valid. Fix or remove the rows marked below.");
    } catch (err) {
      setError(messageOf(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <Page title="Try Your Own Data"
          subtitle="Enter incidents or upload a CSV. The trained models predict risk and response time, then plan priorities and safer routes.">
      <ErrorBanner message={error} onClose={() => setError(null)} />
      {notice && <p role="status" className="mb-4 rounded-lg border-2 border-text px-4 py-3 font-medium">{notice}</p>}
      <fieldset disabled={loading} className="min-w-0 space-y-14">
        <div className="grid gap-x-10 gap-y-12 xl:grid-cols-2">
          <InputCard onAdd={addRows} onNotice={setNotice} onError={setError} />
          <MapCard layouts={layouts} layoutName={layoutName} chooseLayout={chooseLayout} grid={grid} start={start} rows={rows} locked={loading}
                   setGrid={(next) => { changed(); setGrid(next); }} setStart={(next) => { changed(); setStart(next); }} />
        </div>
        <RowsCard rows={rows} problems={problems}
                  onRemove={(index) => { changed(); setRows(rows.filter((_, i) => i !== index)); }}
                  onEdit={(index, field, value) => { changed(); setRows(rows.map((row, i) => (i === index ? { ...row, [field]: value } : row))); }}
                  onClear={() => { changed(); setNotice(null); setRows([]); }}
                  onExample={() => addRows(EXAMPLES)}
                  action={<Button variant="primary" loading={loading} disabled={!rows.length || !grid} icon={<Sparkles className="h-4 w-4" />} onClick={analyse}>
                    Analyse {rows.length || ""}</Button>} />
      </fieldset>
      {result?.plan && <Results key={result.incidents.map((i) => i.incident_id).join("|")} data={result} plan={result.plan} />}
    </Page>
  );
}

// ---------------------------------------------------------------- input

function InputCard({ onAdd, onNotice, onError }: {
  onAdd: (rows: Row[]) => boolean; onNotice: (text: string | null) => void; onError: (text: string | null) => void;
}) {
  const [mode, setMode] = useState<"form" | "csv">("form");
  const [form, setForm] = useState<Row>(BLANK);
  const [dragging, setDragging] = useState(false);
  const [reading, setReading] = useState(false);
  const file = useRef<HTMLInputElement>(null);
  const input = "focus-ring h-9 w-full rounded-lg border border-border bg-surface-2 px-2.5 text-sm";

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (onAdd([form])) setForm({ ...form, incident_id: "" });
  };

  const upload = async (picked: File | undefined) => {
    if (!picked) return;
    onNotice(null);
    onError(null);
    if (!/\.csv$/i.test(picked.name)) return onError("Please choose a .csv file.");
    if (picked.size > 900_000) return onError("The CSV file is too large (limit 900 KB).");
    setReading(true);
    try {
      const data = await api.post<{ rows: Row[]; errors: Problem[]; ignored_columns: string[] }>("/api/user-data/parse", { text: await picked.text() });
      if (!onAdd(data.rows)) return;
      const badRows = new Set(data.errors.map((e) => e.row)).size;
      const parts = [`Added ${data.rows.length} incident(s) from ${picked.name}.`];
      if (badRows) parts.push(`${badRows} need fixing: click Analyse to mark them.`);
      if (data.ignored_columns.length) parts.push(`Ignored columns: ${data.ignored_columns.join(", ")}.`);
      onNotice(parts.join(" "));
    } catch (err) {
      onError(messageOf(err));
    } finally {
      setReading(false);
      if (file.current) file.current.value = "";
    }
  };

  const drop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    upload(event.dataTransfer.files[0]);
  };

  return (
    <Card title="1. Add incidents" action={<Tabs id="input-mode" size="sm" value={mode} onChange={setMode}
          options={[{ value: "form", label: "Form" }, { value: "csv", label: "CSV upload" }]} />}>
      {mode === "form" ? (
        <form onSubmit={submit} className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          <label className="text-sm text-muted">Name (optional)
            <input className={`${input} mt-1 text-text`} maxLength={40} value={form.incident_id} onChange={(e) => setForm({ ...form, incident_id: e.target.value })} />
          </label>
          <label className="text-sm text-muted">Type
            <select className={`${input} mt-1 text-text`} value={form.incident_type} onChange={(e) => setForm({ ...form, incident_type: e.target.value })}>
              {TYPES.map((type) => <option key={type} value={type}>{type.replace("_", " ")}</option>)}
            </select>
          </label>
          {NUMBERS.map(({ field, label, min, max, step, optional }) => (
            <label key={field} className="text-sm text-muted">{label} <span className="num">({min}–{max}{optional ? ", optional" : ""})</span>
              <input type="number" className={`${input} num mt-1 text-text`} min={min} max={max} step={step} required={!optional}
                     value={form[field]} onChange={(e) => setForm({ ...form, [field]: e.target.value })} />
            </label>
          ))}
          <div className="col-span-2 flex items-end sm:col-span-3">
            <Button type="submit" icon={<Plus className="h-4 w-4" />}>Add incident</Button>
          </div>
        </form>
      ) : (
        <div className="space-y-3">
          <div onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={drop}
               className={`flex flex-col items-center gap-3 rounded-card border border-dashed px-4 py-10 text-center transition-colors ${dragging ? "border-accent bg-accent/10" : "border-border"}`}>
            <FileUp className="h-7 w-7 text-muted" />
            <p className="text-sm text-muted">Drop a CSV file here (up to {MAX_ROWS} incidents)</p>
            <input ref={file} type="file" accept=".csv,text/csv" className="hidden" onChange={(e) => upload(e.target.files?.[0])} />
            <Button loading={reading} onClick={() => file.current?.click()}>Choose file</Button>
          </div>
          <Button size="sm" variant="ghost" icon={<Download className="h-3.5 w-3.5" />}
                  onClick={() => download("incident_template.csv", [[...FIELDS], ...EXAMPLES.map((row) => FIELDS.map((f) => row[f]))])}>
            Download template
          </Button>
          <p className="text-sm text-muted">Name, crowd, equipment and obstruction may be blank. Everything else is required.</p>
        </div>
      )}
    </Card>
  );
}

function MapCard({ layouts, layoutName, chooseLayout, grid, start, rows, locked, setGrid, setStart }: {
  layouts: Record<string, Layout> | null; layoutName: string; chooseLayout: (name: string) => void;
  grid: GridData | null; start: Cell; rows: Row[]; locked: boolean;
  setGrid: (grid: GridData) => void; setStart: (cell: Cell) => void;
}) {
  const colors = useColors();
  const [tool, setTool] = useState<"start" | "wall">("start");
  const click = (cell: Cell) => {
    if (!grid || locked) return;
    const [r, c] = cell;
    if (tool === "start") {
      if (grid[r][c] !== -1) setStart(cell);
      return;
    }
    if (cellKey(cell) === cellKey(start)) return;
    const original = layouts?.[layoutName].grid[r][c] ?? 1;
    setGrid(grid.map((line, i) => line.map((v, j) => (i === r && j === c ? (v === -1 ? (original === -1 ? 1 : original) : -1) : v))));
  };
  const markers: Marker[] = rows
    .map((row) => ({ r: Number(row.grid_y || NaN), c: Number(row.grid_x || NaN), name: row.incident_id }))
    .filter(({ r, c }) => Number.isInteger(r) && Number.isInteger(c) && r >= 0 && r < 15 && c >= 0 && c < 15)
    .map(({ r, c, name }) => ({ cell: [r, c], color: colors.accent, kind: "dot", title: name || "Incident" }));
  return (
    <Card title="2. Set up the station" subtitle="S is the vehicle's start. Dots are your incidents."
          action={layouts && <Tabs id="user-layout" size="sm" value={layoutName} onChange={chooseLayout}
            options={Object.keys(layouts).map((name) => ({ value: name, label: name.replace(" Layout", "") }))} />}>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Tabs id="user-tool" size="sm" value={tool} onChange={setTool} options={[
          { value: "start", label: <span className="inline-flex items-center gap-1"><MapPin className="h-3 w-3" />Move start</span> },
          { value: "wall", label: "Add / remove walls" }]} />
        <Button size="sm" variant="ghost" icon={<Undo2 className="h-3.5 w-3.5" />} onClick={() => chooseLayout(layoutName)}>Reset map</Button>
      </div>
      {grid ? <div className="mx-auto max-w-[420px]"><StationGrid grid={grid} start={start} markers={markers} onCellClick={click} label="Station map for your incidents" /></div>
        : <Skeleton className="aspect-square w-full max-w-[420px]" />}
    </Card>
  );
}

function RowsCard({ rows, problems, onRemove, onEdit, onClear, onExample, action }: {
  rows: Row[]; problems: Problem[]; onRemove: (index: number) => void; onEdit: (index: number, field: Field, value: string) => void;
  onClear: () => void; onExample: () => void; action: React.ReactNode;
}) {
  const badCells = new Set(problems.map((p) => `${p.row}-${p.field}`));
  return (
    <Card title="3. Check and analyse" subtitle="Click any value to edit it."
          action={<div className="flex flex-wrap justify-end gap-2">
            {rows.length > 0 && <Button size="sm" variant="ghost" icon={<Trash2 className="h-3.5 w-3.5" />} onClick={onClear}>Clear all</Button>}
            {action}
          </div>}>
      {rows.length === 0 ? (
        <EmptyState icon={<Plus className="h-5 w-5" />} title="No incidents yet" text="Add one with the form, upload a CSV, or start with three examples."
                    action={<Button onClick={onExample}>Load examples</Button>} />
      ) : (
        <div className="max-h-96 overflow-auto">
          <table className="w-full min-w-[980px] text-sm">
            <thead className="sticky top-0 z-10 bg-surface text-left text-muted">
              <tr><th className="p-2">#</th>{FIELDS.map((f) => <th key={f} className="p-2 font-medium">{f.replace(/_/g, " ")}</th>)}<th /></tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={i} className="border-t border-border/50">
                  <td className="num p-2 text-muted">{i + 1}</td>
                  {FIELDS.map((field) => (
                    <td key={field} className="p-1">
                      <input value={row[field]} onChange={(e) => onEdit(i, field, e.target.value)} aria-label={`Row ${i + 1} ${field}`}
                             aria-invalid={badCells.has(`${i + 1}-${field}`)} placeholder={field === "incident_id" ? "auto" : "—"}
                             className={`focus-ring num h-8 w-full rounded-md border px-2 ${badCells.has(`${i + 1}-${field}`) ? "border-danger bg-danger/10" : "border-transparent bg-surface-2"}`} />
                    </td>
                  ))}
                  <td className="p-1"><button type="button" onClick={() => onRemove(i)} aria-label={`Remove row ${i + 1}`} className="focus-ring rounded p-1.5 text-muted hover:text-danger"><Trash2 className="h-4 w-4" /></button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {problems.length > 0 && (
        <div role="alert" className="mt-4 rounded-lg border border-danger/40 bg-danger/5 p-3 text-sm text-danger">
          <p className="mb-1 flex items-center gap-1.5 font-semibold"><OctagonAlert className="h-4 w-4" />These rows were left out of the results:</p>
          <ul className="max-h-40 list-inside list-disc overflow-auto">{problems.map((p, i) => <li key={i}>{p.message}</li>)}</ul>
        </div>
      )}
    </Card>
  );
}

// ---------------------------------------------------------------- results

function Results({ data, plan }: { data: Analysis; plan: NonNullable<Analysis["plan"]> }) {
  const colors = useColors();
  const reachable = data.incidents.filter((i) => i.reachable);
  const [selectedId, setSelectedId] = useState(data.incidents[0].incident_id);
  const [route, setRoute] = useState<"risk" | "normal">("risk");
  const selected = data.incidents.find((i) => i.incident_id === selectedId) ?? null;
  const path = selectedId === TOUR ? plan.tour : (route === "risk" ? selected?.risk_aware : selected?.normal)?.path ?? null;
  const total = path ? path.length - 1 : 0;
  const playback = usePlayback(total, 130);

  useEffect(() => {
    playback.reset();
    const id = window.setTimeout(playback.play, 350);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId, route]);

  const riskColor: Record<RiskName, string> = { Low: colors.success, Medium: colors.warning, High: colors.danger };
  const exportResults = () => download("incident_results.csv", [
    ["priority", "incident_id", "incident_type", "grid_x", "grid_y", "predicted_risk", "probability_low", "probability_medium", "probability_high",
      "predicted_response_time_min", "cluster", "reachable", "reason", "priority_score", "normal_route_cost", "normal_predicted_time", "normal_predicted_risk",
      "risk_aware_route_cost", "risk_aware_predicted_time", "risk_aware_predicted_risk", "risk_aware_path_row_column"],
    ...data.incidents.map((i) => [i.priority, i.incident_id, i.incident_type, i.cell[1], i.cell[0], i.risk_level,
      ...RISK_NAMES.map((n) => i.probabilities[n].toFixed(4)), i.predicted_response_time.toFixed(2), i.cluster, i.reachable ? "yes" : "no", i.reason, i.score,
      i.normal?.travel_cost, i.normal?.predicted_time, i.normal?.predicted_risk, i.risk_aware?.travel_cost, i.risk_aware?.predicted_time,
      i.risk_aware?.predicted_risk, i.risk_aware ? pathCells(i.risk_aware.path) : ""]),
  ]);

  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="mt-14 space-y-12" aria-live="polite">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Analysed" value={`${data.incidents.length} of ${data.submitted}`} />
        <Stat label="Predicted High risk" value={data.incidents.filter((i) => i.risk_level === "High").length} tone="danger" />
        <Stat label="Reachable" value={`${reachable.length} of ${data.incidents.length}`} />
        <Stat label="Cost to visit all reachable" value={fmt(plan.tour_cost, 0)} />
      </div>
      {data.warnings.length > 0 && (
        <div className="rounded-lg border border-warning/40 bg-warning/5 p-3 text-sm text-warning">
          <ul className="max-h-32 overflow-auto">{data.warnings.map((w, i) => <li key={i} className="flex gap-1.5"><TriangleAlert className="h-3.5 w-3.5 shrink-0" />{w.message}</li>)}</ul>
        </div>
      )}
      <div className="grid gap-x-10 gap-y-12 xl:grid-cols-2">
        <Card title="Response order" subtitle="Select an incident to see its routes."
              action={<Button size="sm" icon={<Download className="h-3.5 w-3.5" />} onClick={exportResults}>Download results</Button>}>
          <ol className="max-h-[680px] space-y-2 overflow-auto p-1">
            {data.incidents.map((incident) => (
              <li key={incident.incident_id}>
                <button type="button" onClick={() => setSelectedId(incident.incident_id)} aria-pressed={selectedId === incident.incident_id}
                        className={`focus-ring w-full rounded-lg border p-3 text-left transition-colors ${selectedId === incident.incident_id ? "border-accent/60 bg-accent/10" : "border-border bg-surface-2/50 hover:border-accent/40"}`}>
                  <div className="flex items-center gap-3">
                    <span className="num grid h-7 w-7 shrink-0 place-items-center bg-accent text-sm font-semibold text-onaccent">{incident.priority ?? "–"}</span>
                    <span className="min-w-0 flex-1 truncate text-sm font-medium">{incident.incident_id}
                      <span className="ml-2 text-sm font-normal text-muted">{incident.incident_type.replace("_", " ")}</span></span>
                    <RiskBadge risk={incident.risk_level} size="sm" />
                  </div>
                  <div className="mt-2 flex h-1.5 overflow-hidden bg-surface-2" aria-hidden>
                    {RISK_NAMES.map((n) => <span key={n} style={{ width: `${incident.probabilities[n] * 100}%`, background: riskColor[n] }} />)}
                  </div>
                  <div className="num mt-1.5 flex flex-wrap gap-x-3 text-[13px] text-muted">
                    {RISK_NAMES.map((n) => <span key={n}>{n} {(incident.probabilities[n] * 100).toFixed(1)}%</span>)}
                    <span>· response {incident.predicted_response_time.toFixed(1)} min</span><span>· area {incident.cluster}</span>
                  </div>
                  {!incident.reachable && <p className="mt-1.5 text-sm text-danger">{incident.reason}</p>}
                </button>
              </li>
            ))}
          </ol>
        </Card>

        <Card title={selected ? `Route to ${selected.incident_id}` : "Visit all incidents"}
              subtitle={selected ? "Triangles (Medium) and squares (High) are predicted risk from your other incidents." : "Every reachable incident, in priority order, avoiding predicted risk."}
              action={<div className="flex flex-wrap items-center gap-x-6 gap-y-2">
                {selected && <Tabs id="user-route" size="sm" value={route} onChange={setRoute} options={[{ value: "risk", label: "Risk-aware" }, { value: "normal", label: "Normal" }]} />}
                {reachable.length > 0 && <Button size="sm" variant={selected ? "secondary" : "primary"} onClick={() => setSelectedId(TOUR)}>Visit all</Button>}
              </div>}>
          {selected && !selected.reachable && <div className="mb-3 flex items-center gap-2 rounded-lg bg-danger/10 px-3 py-2 text-sm text-danger"><OctagonAlert className="h-4 w-4" />{selected.reason}</div>}
          <div className="mx-auto max-w-[560px]">
            <StationGrid grid={data.grid} start={data.start} goal={selected?.cell} hazards={plan.predicted_map}
                         vehicle={path ? path[Math.min(playback.step, total)] : null}
                         paths={selected ? [
                           { id: `normal-${selected.incident_id}`, cells: selected.normal?.path, color: colors.accent, dashed: true, width: 4 },
                           { id: `aware-${selected.incident_id}`, cells: selected.risk_aware?.path, color: colors.path, width: 5 },
                         ] : [{ id: "tour", cells: plan.tour, color: colors.path, width: 4, duration: 2 }]}
                         markers={data.incidents.map((i) => ({ cell: i.cell, color: riskColor[i.risk_level], kind: "ring" as const, title: `${i.incident_id}: ${i.risk_level}` }))}
                         label="Routes to your incidents" />
          </div>
          {path && <div className="mt-3"><PlaybackControls playback={playback} total={total} label="Cell" /></div>}
          <div className="mt-3"><GridLegend items={[
            ...(selected ? [{ color: colors.accent, label: "Normal route", shape: "dashed" as const }] : []),
            { color: colors.path, label: "Risk-aware route", shape: "line" },
            { color: colors.warning, label: "Predicted Medium", shape: "triangle" as const }, { color: colors.danger, label: "Predicted High" },
          ]} /></div>
          {selected?.normal && selected.risk_aware && (
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              {([["Normal", selected.normal], ["Risk-aware", selected.risk_aware]] as const).map(([label, info]) => (
                <div key={label} className="rounded-lg border border-border p-3">
                  <div className="mb-2 flex justify-between text-sm font-medium">{label}<Badge tone={info.predicted_risk ? "danger" : "success"}>{info.predicted_risk} predicted risk</Badge></div>
                  <div className="grid grid-cols-2 gap-2"><Stat label="Travel cost" value={fmt(info.travel_cost, 0)} /><Stat label="With risk delays" value={fmt(info.predicted_time, 0)} /></div>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
      <p className="text-sm text-muted">These are predictions from models trained on 360 past incidents. Your data has no true risk labels, so accuracy cannot be measured here. Nothing is retrained. Route time (travel cost plus predicted-risk delays) is separate from the KNN response-time prediction.</p>
    </motion.div>
  );
}
