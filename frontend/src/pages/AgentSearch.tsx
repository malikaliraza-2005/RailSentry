import { AnimatePresence, motion } from "framer-motion";
import {
  BellRing, Brush, CheckCircle2, Eye, Flag, Hash, MapPin, OctagonAlert, RefreshCw, Search, Trophy, Undo2,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { PlaybackControls } from "../components/PlaybackControls";
import { GridLegend, StationGrid } from "../components/StationGrid";
import { Badge, Button, Card, ErrorBanner, Page, Skeleton, Stat, Tabs, fmt } from "../components/ui";
import { api, cellKey, type Cell, type GridData, type Layout, type SearchResult } from "../lib/api";
import { useColors } from "../lib/theme";
import { usePlayback } from "../lib/usePlayback";

type Algorithm = "BFS" | "DFS" | "A*";
const ALGORITHMS: Algorithm[] = ["BFS", "DFS", "A*"];
type Tool = "wall" | "start" | "goal";

interface SearchResponse {
  results: Record<Algorithm, SearchResult>;
  best: Algorithm | null;
}

interface TraceStep {
  position_before: Cell;
  position: Cell;
  percept: Record<string, string>;
  action: string;
  replanned: boolean;
  discovered_blocked: Cell[];
  planned_route: Cell[];
  route_cost: number;
}

interface Scenario {
  name: string;
  description: string;
  grid: GridData;
  start: Cell;
  target: Cell;
  trace: TraceStep[];
  path_taken: Cell[];
  result: { success: number; route_cost: number; steps: number; blocked_replans: number; final_action: string; performance: number };
}

export default function AgentSearch() {
  const [section, setSection] = useState<"search" | "agent">("search");
  return (
    <Page
      title="Agent & Search"
      subtitle="Three search methods race for the cheapest route. Then watch the agent find its way seeing one square ahead."
      actions={
        <Tabs id="m1-section" value={section} onChange={setSection}
              options={[{ value: "search", label: "Algorithm race" }, { value: "agent", label: "Agent playback" }]} />
      }
    >
      <AnimatePresence mode="wait">
        <motion.div key={section} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
          {section === "search" ? <AlgorithmRace /> : <AgentPlayback />}
        </motion.div>
      </AnimatePresence>
    </Page>
  );
}

// ---------------------------------------------------------------- search

function AlgorithmRace() {
  const colors = useColors();
  const [layouts, setLayouts] = useState<Record<string, Layout> | null>(null);
  const [layoutName, setLayoutName] = useState("Normal Layout");
  const [grid, setGrid] = useState<GridData | null>(null);
  const [baseGrid, setBaseGrid] = useState<GridData | null>(null);
  const [start, setStart] = useState<Cell>([13, 1]);
  const [goal, setGoal] = useState<Cell>([1, 13]);
  const [tool, setTool] = useState<Tool>("wall");
  const [editing, setEditing] = useState(false);
  const [showCosts, setShowCosts] = useState(false);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const longest = response ? Math.max(...ALGORITHMS.map((a) => response.results[a].explored.length)) : 0;
  const playback = usePlayback(longest, 60);

  const loadLayout = useCallback((all: Record<string, Layout>, name: string) => {
    const layout = all[name];
    setGrid(layout.grid.map((row) => [...row]));
    setBaseGrid(layout.grid.map((row) => row.map((v) => (v === -1 ? 1 : v))));
    setStart(layout.start);
    setGoal(layout.goal);
    setResponse(null);
  }, []);

  useEffect(() => {
    api.get<Record<string, Layout>>("/api/layouts")
      .then((data) => {
        setLayouts(data);
        loadLayout(data, "Normal Layout");
      })
      .catch((err) => setError(err.message));
  }, [loadLayout]);

  const runSearch = useCallback(async () => {
    if (!grid) return;
    setLoading(true);
    setError(null);
    try {
      const data = await api.post<SearchResponse>("/api/search", { grid, start, goal });
      setResponse(data);
      playback.reset();
      window.setTimeout(playback.play, 50);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [grid, start, goal, playback]);

  // Run automatically when a layout is loaded.
  useEffect(() => {
    if (grid && !response && !editing) runSearch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [grid, editing]);

  const onCellClick = (cell: Cell) => {
    if (!grid || !baseGrid || !editing) return;
    const [r, c] = cell;
    const next = grid.map((row) => [...row]);
    if (tool === "wall") {
      if (cellKey(cell) === cellKey(start) || cellKey(cell) === cellKey(goal)) return;
      next[r][c] = next[r][c] === -1 ? baseGrid[r][c] : -1;
    } else {
      if (next[r][c] === -1) next[r][c] = baseGrid[r][c];
      if (tool === "start") setStart(cell);
      else setGoal(cell);
    }
    setGrid(next);
    setResponse(null);
    playback.reset();
  };

  const layoutOptions = layouts ? Object.keys(layouts).map((name) => ({ value: name, label: name.replace(" Layout", "") })) : [];
  const algoColors: Record<Algorithm, string> = { BFS: colors.accent, DFS: colors.warning, "A*": colors.path };

  return (
    <div className="space-y-14">
      <ErrorBanner message={error} onClose={() => setError(null)} />
      <Card
        title="Station layout"
        subtitle={editing ? "Click cells to edit. Walls toggle on and off; start and goal move." : "Choose a layout, or edit it to build your own."}
        action={
          <div className="flex flex-wrap items-center gap-2">
            {layouts && (
              <Tabs id="layout" size="sm" value={layoutName} options={layoutOptions}
                    onChange={(name) => { setLayoutName(name); setEditing(false); loadLayout(layouts, name); }} />
            )}
          </div>
        }
      >
        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm" variant={editing ? "primary" : "secondary"} icon={<Brush className="h-3.5 w-3.5" />}
                  onClick={() => setEditing((e) => !e)}>
            {editing ? "Done editing" : "Edit grid"}
          </Button>
          {editing && (
            <Tabs id="tool" size="sm" value={tool} onChange={setTool} options={[
              { value: "wall", label: <span className="inline-flex items-center gap-1"><Hash className="h-3 w-3" />Wall</span> },
              { value: "start", label: <span className="inline-flex items-center gap-1"><MapPin className="h-3 w-3" />Start</span> },
              { value: "goal", label: <span className="inline-flex items-center gap-1"><Flag className="h-3 w-3" />Goal</span> },
            ]} />
          )}
          {editing && layouts && (
            <Button size="sm" variant="ghost" icon={<Undo2 className="h-3.5 w-3.5" />} onClick={() => loadLayout(layouts, layoutName)}>Reset layout</Button>
          )}
          <Button size="sm" variant="ghost" icon={<Eye className="h-3.5 w-3.5" />} onClick={() => setShowCosts((s) => !s)}>
            {showCosts ? "Hide costs" : "Show costs"}
          </Button>
          <Button size="sm" variant="primary" className="ml-auto" icon={<Search className="h-3.5 w-3.5" />} loading={loading} onClick={runSearch} disabled={!grid}>
            Run search
          </Button>
        </div>

        {editing && grid && (
          <div className="mx-auto mt-4 max-w-md">
            <StationGrid grid={grid} start={start} goal={goal} onCellClick={onCellClick} showCosts={showCosts} label="Editable station grid" />
          </div>
        )}
      </Card>

      {!editing && (
        <Card
          title="Algorithm race"
          subtitle="Explored cells light up in the order each algorithm expanded them; the final route then draws itself."
          action={response?.best && playback.done ? <Badge tone="success"><Trophy className="h-3 w-3" /> Best: {response.best}</Badge> : undefined}
        >
          {response && <div className="mb-4"><PlaybackControls playback={playback} total={longest} label="Nodes" /></div>}
          <div className="grid gap-4 lg:grid-cols-3">
            {ALGORITHMS.map((algorithm) => {
              const result = response?.results[algorithm];
              const finished = result ? playback.step >= result.explored.length : false;
              return (
                <div key={algorithm} className={`rounded-card border p-3 transition-colors ${
                  finished && response?.best === algorithm ? "border-success/60 bg-success/5" : "border-border bg-surface-2/40"}`}>
                  <div className="mb-2 flex items-center justify-between">
                    <span className="inline-flex items-center gap-2 text-sm font-semibold">
                      <span className="h-3 w-3" style={{ background: algoColors[algorithm] }} />{algorithm}
                    </span>
                    <span className="num text-sm text-muted">
                      {result ? `${Math.min(playback.step, result.explored.length)} / ${result.nodes_expanded} nodes` : ""}
                    </span>
                  </div>
                  {grid && result ? (
                    <StationGrid
                      grid={grid} start={start} goal={goal} showCosts={showCosts}
                      explored={result.explored.slice(0, playback.step)}
                      exploredColor={colors.alpha(algorithm === "A*" ? "path" : algorithm === "BFS" ? "accent" : "warning", 0.3)}
                      paths={finished && result.path ? [{ id: algorithm, cells: result.path, color: algoColors[algorithm], width: 6 }] : []}
                      label={`${algorithm} search result`}
                    />
                  ) : <Skeleton className="aspect-square w-full" />}
                  <AnimatePresence>
                    {result && finished && !result.path && (
                      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                                  className="mt-2 flex items-center gap-2 rounded-lg bg-danger/10 px-3 py-2 text-sm text-danger">
                        <OctagonAlert className="h-4 w-4" /> No path exists
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              );
            })}
          </div>

          {response && (
            <div className="mt-5 overflow-x-auto">
              <table className="w-full min-w-[520px] text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-sm text-muted">
                    <th className="py-2 font-medium">Algorithm</th>
                    <th className="py-2 text-right font-medium">Path length</th>
                    <th className="py-2 text-right font-medium">Total cost</th>
                    <th className="py-2 text-right font-medium">Nodes expanded</th>
                    <th className="py-2 text-right font-medium">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {ALGORITHMS.map((algorithm) => {
                    const r = response.results[algorithm];
                    const best = response.best === algorithm;
                    return (
                      <tr key={algorithm} className={`border-b border-border/60 ${best ? "text-success" : ""}`}>
                        <td className="py-2 font-medium">{algorithm} {best && <Trophy className="ml-1 inline h-3.5 w-3.5" />}</td>
                        <td className="num py-2 text-right">{r.path ? r.path_length : "—"}</td>
                        <td className="num py-2 text-right">{r.path ? r.cost.toFixed(0) : "—"}</td>
                        <td className="num py-2 text-right">{r.nodes_expanded}</td>
                        <td className="num py-2 text-right">{r.time_ms.toFixed(3)} ms</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="mt-3 text-sm text-muted">
                {response.best
                  ? `${response.best} has the lowest total route cost on this weighted grid. BFS finds the fewest steps, not the cheapest route; DFS finds any route.`
                  : "None of the algorithms found a path: the goal is unreachable."}
              </p>
            </div>
          )}
        </Card>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- agent

function AgentPlayback() {
  const colors = useColors();
  const [scenarios, setScenarios] = useState<Scenario[] | null>(null);
  const [formula, setFormula] = useState("");
  const [selected, setSelected] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.get<{ scenarios: Scenario[]; performance_formula: string }>("/api/agent/scenarios")
      .then((data) => {
        setScenarios(data.scenarios);
        setFormula(data.performance_formula);
      })
      .catch((err) => setError(err.message));
  }, []);

  const scenario = scenarios?.[selected];
  const playback = usePlayback(scenario?.trace.length ?? 0, 300);

  useEffect(() => {
    playback.reset();
    const id = window.setTimeout(playback.play, 300);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, scenarios]);

  const view = useMemo(() => {
    if (!scenario) return null;
    const step = playback.step;
    const current = step > 0 ? scenario.trace[step - 1] : null;
    const position = current ? current.position : scenario.start;
    const pathSoFar = scenario.path_taken.slice(0, step + 1);
    const discovered = new Set((current?.discovered_blocked ?? []).map(cellKey));
    const unknown = new Set<string>();
    scenario.grid.forEach((row, r) => row.forEach((v, c) => {
      if (v === -1 && !discovered.has(`${r},${c}`)) unknown.add(`${r},${c}`);
    }));
    return { current, position, pathSoFar, unknown };
  }, [scenario, playback.step]);

  if (error) return <ErrorBanner message={error} />;
  if (!scenarios || !scenario || !view) return <Skeleton className="h-[480px]" />;

  const finished = playback.done;
  const final = scenario.result;
  const alert = final.final_action.startsWith("alert");
  const success = final.success === 1;

  return (
    <div className="grid gap-x-10 gap-y-12 lg:grid-cols-[260px_1fr]">
      <div className="space-y-2">
        {scenarios.map((s, index) => (
          <button
            key={s.name}
            onClick={() => setSelected(index)}
            className={`focus-ring relative w-full rounded-lg border px-3 py-2.5 text-left transition-colors ${
              index === selected ? "border-accent/60 bg-accent/10" : "border-border bg-surface hover:border-accent/40"}`}
          >
            <div className="num text-sm text-muted">Scenario {index + 1}</div>
            <div className="text-sm font-medium">{s.name}</div>
          </button>
        ))}
      </div>

      <div className="grid gap-x-10 gap-y-12 xl:grid-cols-[1fr_320px]">
        <Card title={scenario.name} subtitle={scenario.description}>
          <div className="mb-4"><PlaybackControls playback={playback} total={scenario.trace.length} label="Action" /></div>
          <div className="mx-auto max-w-[560px]">
            <StationGrid
              grid={scenario.grid}
              start={scenario.start}
              goal={scenario.target}
              unknownBlocked={view.unknown}
              vehicle={view.position}
              paths={[
                { id: "planned", cells: view.current?.planned_route?.length ? [view.position, ...view.current.planned_route.slice(1)] : null, color: colors.alpha("accent", 0.7), dashed: true, width: 4, duration: 0.4 },
                { id: "taken", cells: view.pathSoFar, color: colors.path, width: 5, duration: 0.25 },
              ]}
              markers={(view.current?.discovered_blocked ?? []).map((cell) => ({
                cell, color: colors.danger, kind: "x" as const, title: "Discovered blocked cell",
              }))}
              label={`Agent scenario: ${scenario.name}`}
            />
          </div>
          <div className="mt-4">
            <GridLegend items={[
              { color: colors.path, label: "Route taken", shape: "line" },
              { color: colors.accent, label: "Current plan", shape: "dashed" },
              { color: colors.danger, label: "Discovered wall" },
              { color: colors.muted, label: "Wall not yet sensed (dashed cell)" },
            ]} />
          </div>
        </Card>

        <div className="space-y-4">
          <Card title="Agent state">
            <div className="grid grid-cols-2 gap-2">
              <Stat label="Position" value={`(${view.position[0]}, ${view.position[1]})`} />
              <Stat label="Route cost" value={view.current ? view.current.route_cost.toFixed(0) : "0"} />
              <Stat label="Re-plans" value={scenario.trace.slice(0, playback.step).filter((t) => t.replanned).length}
                    tone={scenario.trace.slice(0, playback.step).some((t) => t.replanned) ? "warning" : undefined} />
              <Stat label="Walls known" value={view.current?.discovered_blocked.length ?? 0} />
            </div>
            <div className="mt-4">
              <div className="mb-1.5 text-[13px] text-muted">Percept (before acting)</div>
              {view.current ? (
                <ul className="grid grid-cols-2 gap-1.5 text-sm">
                  {Object.entries(view.current.percept).map(([cell, value]) => (
                    <li key={cell} className="flex items-center justify-between rounded-md bg-surface-2 px-2 py-1.5">
                      <span className="num text-muted">({cell.replace(",", ", ")})</span>
                      <span className={value === "blocked" ? "text-danger" : value === "goal" ? "text-warning" : value === "out_of_bounds" ? "text-muted" : "text-success"}>
                        {value.replace("_", " ")}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : <p className="text-sm text-muted">Press play to start.</p>}
            </div>
            <div className="mt-4">
              <div className="mb-1.5 text-[13px] text-muted">Selected action</div>
              <AnimatePresence mode="wait">
                <motion.div
                  key={playback.step}
                  initial={{ opacity: 0, x: 8 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -8 }}
                  transition={{ duration: 0.15 }}
                  className={`flex items-center gap-2 rounded-lg px-3 py-2 text-sm ${
                    view.current?.replanned ? "bg-warning/15 text-warning" : view.current?.action.startsWith("alert") ? "bg-danger/10 text-danger" : "bg-surface-2"}`}
                >
                  {view.current?.replanned && <RefreshCw className="h-4 w-4 shrink-0" />}
                  {view.current?.action ?? "waiting"}
                </motion.div>
              </AnimatePresence>
            </div>
          </Card>

          <AnimatePresence>
            {finished && (
              <motion.div
                initial={{ opacity: 0, y: 10, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0 }}
                className={`border-2 p-5 ${success ? "border-success" : "border-danger"}`}
              >
                <div className={`flex items-center gap-2 font-semibold ${success ? "text-success" : "text-danger"}`}>
                  {success ? <CheckCircle2 className="h-5 w-5" /> : alert ? <BellRing className="h-5 w-5" /> : <OctagonAlert className="h-5 w-5" />}
                  {success ? "Incident location reached" : "Alert raised"}
                </div>
                <p className="mt-1 text-sm text-muted">{final.final_action}</p>
                <div className="mt-3 grid grid-cols-3 gap-2">
                  <Stat label="Steps" value={final.steps} />
                  <Stat label="Re-plans" value={final.blocked_replans} />
                  <Stat label="Score" value={fmt(final.performance, 0)} tone={final.performance >= 0 ? "success" : "danger"} />
                </div>
                <p className="mt-3 text-[13px] text-muted">Performance = {formula}</p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
