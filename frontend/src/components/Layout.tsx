import { motion, useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { api } from "../lib/api";
import { useTheme, type ThemeChoice } from "../lib/theme";

/** The stops on the line, in travelling order. Numbered stops are the project modules. */
export const STOPS: { to: string; label: string; number?: number }[] = [
  { to: "/", label: "Dashboard" },
  { to: "/agent", label: "Agent & Search", number: 1 },
  { to: "/data", label: "Data & ML", number: 2 },
  { to: "/neural", label: "Neural Networks", number: 3 },
  { to: "/system", label: "Smart System", number: 4 },
  { to: "/try", label: "Try Your Own Data" },
  { to: "/logs", label: "Logs" },
];

function Clock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(id);
  }, []);
  const hand = (angle: number, length: number, width: number) => (
    <line x1={20} y1={20} x2={20} y2={20 - length} strokeWidth={width} stroke="currentColor" transform={`rotate(${angle} 20 20)`} />
  );
  return (
    <span className="hidden items-center gap-2.5 sm:flex">
      <svg viewBox="0 0 40 40" className="h-9 w-9" aria-hidden>
        <circle cx={20} cy={20} r={18.5} fill="none" stroke="currentColor" strokeWidth={1.5} />
        {Array.from({ length: 12 }).map((_, i) => (
          <line key={i} x1={20} y1={3.5} x2={20} y2={i % 3 ? 6 : 8.5} stroke="currentColor" strokeWidth={i % 3 ? 1 : 2} transform={`rotate(${i * 30} 20 20)`} />
        ))}
        {hand((now.getHours() % 12) * 30 + now.getMinutes() / 2, 9, 2.6)}
        {hand(now.getMinutes() * 6, 13.5, 2)}
        {hand(now.getSeconds() * 6, 14.5, 0.8)}
      </svg>
      <time className="num font-display text-2xl font-semibold leading-none">
        {now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false })}
      </time>
    </span>
  );
}

function BackendStatus() {
  const [state, setState] = useState<"online" | "busy" | "offline">("offline");
  useEffect(() => {
    let alive = true;
    const check = async () => {
      try {
        const health = await api.get<{ busy: boolean }>("/api/health");
        if (alive) setState(health.busy ? "busy" : "online");
      } catch {
        if (alive) setState("offline");
      }
    };
    check();
    const id = window.setInterval(check, 3000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, []);
  const style = {
    online: { mark: "bg-success", text: "API online" },
    busy: { mark: "bg-warning animate-pulse", text: "Task running" },
    offline: { mark: "bg-danger", text: "API offline" },
  }[state];
  return (
    <span className="flex items-center gap-2 whitespace-nowrap text-sm" role="status">
      <span className={`h-2.5 w-2.5 ${style.mark}`} aria-hidden />
      {style.text}
    </span>
  );
}

function ThemePicker() {
  const { choice, setChoice } = useTheme();
  return (
    <label className="text-sm">
      <span className="sr-only">Theme</span>
      <select
        value={choice}
        onChange={(event) => setChoice(event.target.value as ThemeChoice)}
        className="focus-ring cursor-pointer border-b border-text bg-bg py-1 pr-1 text-text"
      >
        <option value="dark">Dark</option>
        <option value="light">Light</option>
        <option value="system">System</option>
      </select>
    </label>
  );
}

/** Navigation drawn as a railway line: each page is a stop, the train marks where you are. */
function LineNav() {
  const { pathname } = useLocation();
  const reduce = useReducedMotion();
  const scroller = useRef<HTMLElement>(null);
  const found = STOPS.findIndex((stop) => stop.to === pathname);
  const current = found === -1 ? 0 : found;

  useEffect(() => {
    const nav = scroller.current;
    const link = nav?.querySelector<HTMLElement>('[aria-current="page"]');
    if (nav && link) nav.scrollTo({ left: link.offsetLeft - nav.clientWidth / 2 + link.clientWidth / 2, behavior: reduce ? "auto" : "smooth" });
  }, [pathname, reduce]);

  return (
    <nav ref={scroller} aria-label="Main" className="line-nav relative overflow-x-auto border-b border-border bg-bg">
      <ol className="mx-auto grid w-max min-w-full max-w-[1400px] grid-cols-7 px-2 sm:px-4">
        {STOPS.map((stop, index) => (
          <li key={stop.to} className="relative min-w-[7.6rem]">
            {/* track to the previous and next stop */}
            {index > 0 && <span className={`absolute left-0 right-1/2 top-[25px] h-[5px] ${index <= current ? "bg-text" : "bg-border"}`} />}
            {index < STOPS.length - 1 && <span className={`absolute left-1/2 right-0 top-[25px] h-[5px] ${index < current ? "bg-text" : "bg-border"}`} />}
            <NavLink to={stop.to} end={stop.to === "/"} className="focus-ring group relative flex flex-col items-center pb-3 pt-4 text-center">
              <span className={`relative grid h-[23px] w-[23px] place-items-center rounded-full border-[3px] bg-bg text-[11px] font-bold leading-none transition-transform group-hover:scale-110 ${index <= current ? "border-text" : "border-border"}`}>
                {stop.number}
                {index === current && (
                  <motion.span
                    layoutId="train"
                    className="absolute -left-[17px] top-[-3px] z-10 flex h-[23px] w-[51px] items-center justify-around bg-accent px-1.5"
                    style={{ borderRadius: 3 }}
                    transition={reduce ? { duration: 0 } : { duration: 0.55, ease: [0.65, 0, 0.25, 1] }}
                    aria-hidden
                  >
                    {[0, 1, 2].map((i) => <i key={i} className="h-2 w-2.5 bg-onaccent" />)}
                  </motion.span>
                )}
              </span>
              <span className={`mt-2 whitespace-nowrap font-display text-[17px] leading-none ${index === current ? "font-semibold text-text" : "font-medium text-muted group-hover:text-text"}`}>
                {stop.label}
              </span>
            </NavLink>
          </li>
        ))}
      </ol>
    </nav>
  );
}

export function Layout({ children }: { children: ReactNode }) {
  const { pathname } = useLocation();
  useEffect(() => { window.scrollTo(0, 0); }, [pathname]);
  return (
    <div className="min-h-screen">
      <header className="mx-auto flex max-w-[1400px] items-center gap-5 px-4 py-4 sm:px-8">
        <NavLink to="/" className="focus-ring mr-auto leading-none">
          <span className="block font-display text-[26px] font-bold leading-none">RailSentry</span>
          <span className="mt-1 hidden text-sm text-muted sm:block">Station incident response</span>
        </NavLink>
        <BackendStatus />
        <ThemePicker />
        <Clock />
      </header>
      <div className="sticky top-0 z-30"><LineNav /></div>
      <main>{children}</main>
    </div>
  );
}
