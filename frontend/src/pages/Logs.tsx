import { ArrowDownToLine, Pause, Play, Search, Trash2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Button, Card, ErrorBanner, Page } from "../components/ui";
import { api } from "../lib/api";

interface Line {
  id: number;
  time: number;
  text: string;
}

export default function Logs() {
  const [lines, setLines] = useState<Line[]>([]);
  const [paused, setPaused] = useState(false);
  const [follow, setFollow] = useState(true);
  const [filter, setFilter] = useState("");
  const [error, setError] = useState<string | null>(null);
  const lastId = useRef(0);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (paused) return;
    let alive = true;
    const tick = async () => {
      try {
        const data = await api.get<{ lines: Line[] }>(`/api/logs?after=${lastId.current}`);
        if (!alive) return;
        if (data.lines.length) {
          lastId.current = data.lines[data.lines.length - 1].id;
          setLines((current) => [...current, ...data.lines].slice(-2000));
        }
        setError(null);
      } catch (err) {
        if (alive) setError(err instanceof Error ? err.message : String(err));
      }
    };
    tick();
    const id = window.setInterval(tick, 1000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [paused]);

  const shown = useMemo(() => {
    const q = filter.trim().toLowerCase();
    return q ? lines.filter((l) => l.text.toLowerCase().includes(q)) : lines;
  }, [lines, filter]);

  useEffect(() => {
    if (follow && box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [shown, follow]);

  const tone = (text: string) =>
    /error|traceback|exception/i.test(text) ? "text-danger"
      : /warning/i.test(text) ? "text-warning"
      : /completed|best|saved/i.test(text) ? "text-success"
      : "text-text/90";

  return (
    <Page title="Logs" subtitle="Live output from the Python modules, useful while testing.">
      <ErrorBanner message={error} />
      <Card>
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <label className="relative flex-1 sm:max-w-xs">
            <Search className="pointer-events-none absolute left-2.5 top-2.5 h-4 w-4 text-muted" />
            <input value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Filter lines…" aria-label="Filter log lines"
                   className="focus-ring h-9 w-full rounded-lg border border-border bg-surface-2 pl-8 pr-3 text-sm placeholder:text-muted" />
          </label>
          <Button size="sm" icon={paused ? <Play className="h-3.5 w-3.5" /> : <Pause className="h-3.5 w-3.5" />} onClick={() => setPaused((p) => !p)}>
            {paused ? "Resume" : "Pause"}
          </Button>
          <Button size="sm" variant={follow ? "primary" : "secondary"} icon={<ArrowDownToLine className="h-3.5 w-3.5" />} onClick={() => setFollow((f) => !f)}>
            Auto-scroll
          </Button>
          <Button size="sm" variant="ghost" icon={<Trash2 className="h-3.5 w-3.5" />} onClick={() => setLines([])}>Clear</Button>
          <span className="num ml-auto text-sm text-muted">{shown.length} lines</span>
        </div>
        <div ref={box} className="num h-[65vh] overflow-auto rounded-lg border border-border bg-bg p-3 text-[13px] leading-relaxed" role="log" aria-live="polite">
          {shown.length === 0 ? (
            <p className="text-muted">No output yet. Run a module and its output appears here.</p>
          ) : shown.map((line) => (
            <div key={line.id} className="flex gap-3 whitespace-pre-wrap">
              <span className="shrink-0 select-none text-muted/70">{new Date(line.time * 1000).toLocaleTimeString()}</span>
              <span className={tone(line.text)}>{line.text || " "}</span>
            </div>
          ))}
        </div>
      </Card>
    </Page>
  );
}
