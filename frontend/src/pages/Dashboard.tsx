import { Link } from "react-router-dom";
import { Button, Card, ErrorBanner, Flaps, KpiCard, Page, ProgressBar, Skeleton } from "../components/ui";
import { RISK_NAMES } from "../lib/api";
import { useStore } from "../lib/store";
import { useJob } from "../lib/useJob";

const MODULES = [
  { key: "module1", label: "Agent search", to: "/agent", text: "Find a route with BFS, DFS and A*" },
  { key: "module2", label: "Data and ML", to: "/data", text: "Clean the records, group locations, tune KNN" },
  { key: "module3", label: "Neural nets", to: "/neural", text: "Train two networks to predict risk" },
  { key: "module4", label: "Smart system", to: "/system", text: "Route the vehicle around predicted danger" },
];
const RISK_FILL = ["bg-success", "bg-warning", "bg-danger"];

export default function Dashboard() {
  const { overview, overviewError, refreshOverview, results } = useStore();
  const demo = useJob("full_demo", () => refreshOverview(true));

  const statusOf = (key: string) => {
    const last = demo.events.filter((event) => event.stage === key).pop();
    if (demo.running) return last ? (last.status === "done" ? "Done" : "Running") : "Waiting";
    if (!overview) return "";
    return key === "module1" || overview.results[key] ? "Ready" : "Not run";
  };

  const module3 = results.module3 ?? overview?.results.module3;
  const best = module3?.comparison?.find((row: { Model: string }) => row.Model === module3.best_ann);
  const mae = results.module2?.knn_regressor?.mae ?? overview?.results.module2?.knn_regressor?.mae;
  const summary = overview?.results.module4?.summary as { "Layout name": string; Mode: string; "Risk encountered": number }[] | undefined;
  const riskIn = (mode: string) => summary?.find((row) => row["Layout name"] === "Normal Layout" && row.Mode === mode)?.["Risk encountered"];
  const without = riskIn("Without predictions");
  const using = riskIn("With predictions");
  const avoided = without && using !== undefined ? (1 - using / without) * 100 : null;

  const types = overview ? Object.entries(overview.dataset.incident_types) : [];
  const mostTypes = Math.max(1, ...types.map(([, count]) => count));

  return (
    <Page title="Dashboard" subtitle="Predicts how dangerous a station incident is, then sends the response vehicle by the safest route."
          actions={<Button variant="primary" loading={demo.running} onClick={() => demo.start()}>{demo.running ? "Running all modules" : "Run full demo"}</Button>}>
      <ErrorBanner message={overviewError ?? demo.error} onClose={demo.error ? demo.clearError : undefined} />

      <section className="bg-board p-3 text-[#f6f1df] sm:p-5" aria-label="Project modules">
        <ul>
          {MODULES.map((module, index) => {
            const status = statusOf(module.key);
            return (
              <li key={module.key} className="border-b border-white/15 last:border-0">
                <Link to={module.to} className="focus-ring group grid grid-cols-[2rem_1fr_auto] items-center gap-x-3 py-3 sm:gap-x-6 lg:grid-cols-[3rem_19rem_1fr_auto]">
                  <span className="font-display text-3xl font-bold text-flap sm:text-4xl">{index + 1}</span>
                  <span>
                    <span className="sr-only">{module.label}</span>
                    <span aria-hidden className="font-display text-2xl font-semibold sm:hidden">{module.label}</span>
                    <span aria-hidden className="hidden sm:block"><Flaps text={module.label} size="27px" /></span>
                  </span>
                  <span className="col-start-2 row-start-2 mt-1 text-sm text-white/70 group-hover:text-flap lg:col-start-3 lg:row-start-1 lg:mt-0 lg:text-lg">{module.text}</span>
                  <span className="col-start-3 row-start-1 lg:col-start-4">
                    <span className="sr-only">{status}</span>
                    <span aria-hidden className="font-display text-xl font-semibold text-flap sm:hidden">{status}</span>
                    <span aria-hidden className="hidden sm:block"><Flaps text={status || "-"} padTo={7} size="24px" /></span>
                  </span>
                </Link>
              </li>
            );
          })}
        </ul>
      </section>
      {(demo.running || demo.job?.status === "done") && (
        <div className="mt-5"><ProgressBar value={demo.job?.progress ?? 0} label={demo.job?.status === "done" ? "All modules finished" : demo.job?.step || "Starting"} /></div>
      )}

      <div className="mt-14 grid gap-x-8 gap-y-8 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard label="Incidents on record" value={overview?.dataset.records} hint={overview && `${overview.dataset.average_response_time.toFixed(1)} minutes to respond, on average`} />
        <KpiCard label="Risk predicted correctly" value={best ? best.accuracy * 100 : null} decimals={1} suffix="%" hint={best ? `${module3.best_ann}, on unseen incidents` : "Train Module 3 first"} />
        <KpiCard label="Response-time error" value={mae ?? null} decimals={2} suffix="min" hint={mae ? "Average miss, KNN regressor" : "Run Module 2 first"} />
        <KpiCard label="Risk avoided" value={avoided} suffix="%" hint={avoided !== null ? "Normal layout, with predictions on" : "Run Module 4 first"} />
      </div>

      <div className="mt-16 grid gap-x-12 gap-y-14 lg:grid-cols-[1fr_1.4fr]">
        <Card title="How serious are they?">
          {overview ? (
            <>
              <div className="flex h-24 w-full" aria-hidden>
                {RISK_NAMES.map((name, i) => <span key={name} className={RISK_FILL[i]} style={{ width: `${(overview.dataset.risk_counts[name] / overview.dataset.records) * 100}%` }} />)}
              </div>
              <dl className="mt-3 divide-y divide-border">
                {RISK_NAMES.map((name, i) => (
                  <div key={name} className="flex items-baseline gap-3 py-2">
                    <span className={`h-3 w-3 ${RISK_FILL[i]}`} aria-hidden />
                    <dt className="flex-1">{name} risk</dt>
                    <dd className="num font-display text-2xl font-semibold">{overview.dataset.risk_counts[name]}</dd>
                  </div>
                ))}
              </dl>
            </>
          ) : <Skeleton className="h-56" />}
        </Card>
        <Card title="What goes wrong">
          {overview ? (
            <dl className="space-y-3">
              {types.map(([name, count]) => (
                <div key={name} className="grid grid-cols-[9rem_1fr_2.5rem] items-center gap-3">
                  <dt>{name.replace("_", " ")}</dt>
                  <dd className="h-6" aria-hidden><span className="block h-full bg-text" style={{ width: `${(count / mostTypes) * 100}%` }} /></dd>
                  <dd className="num text-right font-display text-2xl font-semibold leading-none">{count}</dd>
                </div>
              ))}
            </dl>
          ) : <Skeleton className="h-56" />}
        </Card>
      </div>

      {overview && (
        <Card title="Saved models" className="mt-16">
          <ul className="grid gap-x-12 sm:grid-cols-2">
            {Object.entries(overview.models).map(([name, exists]) => (
              <li key={name} className="flex justify-between gap-4 border-b border-border py-2">
                <span>{name}</span>
                <span className={exists ? "text-muted" : "font-semibold text-danger"}>{exists ? "Saved" : "Missing"}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </Page>
  );
}
