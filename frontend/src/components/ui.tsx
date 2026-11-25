import { motion, useReducedMotion } from "framer-motion";
import { AlertTriangle, CheckCircle2, Loader2, OctagonAlert, X } from "lucide-react";
import { useId, type ButtonHTMLAttributes, type ReactNode } from "react";
import { useLocation } from "react-router-dom";
import { STOPS } from "./Layout";
import RollCount from "./reactbits/CountUp";
import SplitFlapText from "./reactbits/SplitFlapText";

// ---------- departure-board lettering ----------

export function Flaps({ text, size, padTo, className = "" }: { text: string; size: string; padTo?: number; className?: string }) {
  return (
    <SplitFlapText text={text.toUpperCase()} padTo={padTo ?? text.length} loop={false} intro fontSize={size} gap={2} tileRadius={1}
                   tileColor="#1d1e1b" textColor="#f6f1df" flipsPerChar={3} flipDuration={0.08} stagger={0.02} glare={0} depth={0.3}
                   className={className} />
  );
}

// ---------- layout ----------

export function Page({ title, subtitle, actions, children }: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  const { pathname } = useLocation();
  const number = STOPS.find((stop) => stop.to === pathname)?.number;
  return (
    <div className="mx-auto w-full max-w-[1400px] px-4 pb-24 pt-8 sm:px-8">
      <div className="mb-10 grid items-end gap-5 lg:grid-cols-[1fr_auto]">
        <div className="min-w-0">
          <h1 className="inline-flex max-w-full items-stretch bg-board" aria-label={title}>
            {number && <span aria-hidden className="grid place-items-center bg-flap px-3 font-display text-3xl font-bold text-board sm:px-5 sm:text-5xl">{number}</span>}
            <span aria-hidden className="p-2 sm:p-2.5">
              <Flaps key={title} text={title} size={`clamp(13px, min(44px, calc((100vw - 110px) / ${(title.length * 0.88).toFixed(1)})), 44px)`} />
            </span>
          </h1>
          {subtitle && <p className="mt-4 max-w-2xl text-[17px] leading-snug text-muted">{subtitle}</p>}
        </div>
        {actions && <div className="flex flex-wrap items-center gap-3">{actions}</div>}
      </div>
      {children}
    </div>
  );
}

export function Card({ title, subtitle, action, children, className = "" }: {
  title?: ReactNode;
  subtitle?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  delay?: number;
}) {
  return (
    <section className={`card min-w-0 pt-3 ${className}`}>
      {(title || action) && (
        <div className="mb-5 flex flex-wrap items-start justify-between gap-x-6 gap-y-3">
          <div className="min-w-0 flex-1 basis-56">
            {title && <h2 className="font-display text-2xl font-semibold leading-tight">{title}</h2>}
            {subtitle && <p className="mt-1 max-w-2xl text-sm leading-snug text-muted">{subtitle}</p>}
          </div>
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

// ---------- buttons ----------

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost";
  loading?: boolean;
  icon?: ReactNode;
  size?: "sm" | "md";
};

export function Button({ variant = "secondary", loading, icon, size = "md", children, className = "", disabled, type = "button", ...rest }: ButtonProps) {
  const styles = {
    primary: "bg-accent text-onaccent border-2 border-accent hover:bg-bg hover:text-text",
    secondary: "border-2 border-text text-text hover:bg-text hover:text-bg",
    ghost: "border-2 border-transparent text-text underline decoration-1 underline-offset-4 hover:decoration-[3px]",
  }[variant];
  const sizing = size === "sm" ? "min-h-8 px-2.5 py-1 text-sm" : "min-h-11 px-4 py-2 text-base";
  return (
    <button
      type={type}
      className={`focus-ring inline-flex items-center justify-center gap-2 whitespace-nowrap font-semibold transition-colors duration-100 active:translate-y-px disabled:pointer-events-none disabled:opacity-45 ${sizing} ${styles} ${className}`}
      disabled={disabled || loading}
      {...rest}
    >
      {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : icon}
      {children}
    </button>
  );
}

// ---------- tabs (a ruled line that slides under the current choice) ----------

export function Tabs<T extends string>({ options, value, onChange, id, size = "md" }: {
  options: { value: T; label: ReactNode }[];
  value: T;
  onChange: (value: T) => void;
  id: string;
  size?: "sm" | "md";
}) {
  const reduce = useReducedMotion();
  const unique = useId();
  return (
    <div role="tablist" className="inline-flex max-w-full flex-wrap gap-x-5">
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => onChange(option.value)}
            className={`focus-ring relative whitespace-nowrap border-b-2 border-transparent pb-1.5 pt-1 ${size === "sm" ? "text-sm" : "text-base"} ${active ? "font-semibold text-text" : "font-medium text-muted hover:text-text"}`}
          >
            {active && (
              <motion.span layoutId={`tab-${id}-${unique}`} className="absolute inset-x-0 -bottom-[2px] h-1 bg-accent"
                           transition={reduce ? { duration: 0 } : { duration: 0.25, ease: [0.65, 0, 0.25, 1] }} />
            )}
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

// ---------- badges ----------

const RISK_STYLE = {
  Low: { cls: "border-success text-success", Icon: CheckCircle2 },
  Medium: { cls: "border-warning text-warning", Icon: AlertTriangle },
  High: { cls: "border-danger bg-danger text-bg", Icon: OctagonAlert },
} as const;

/** Risk is shown with colour, an icon and a label (never colour alone). */
export function RiskBadge({ risk, size = "md" }: { risk: string; size?: "sm" | "md" }) {
  const style = RISK_STYLE[risk as keyof typeof RISK_STYLE] ?? RISK_STYLE.Low;
  const Icon = style.Icon;
  return (
    <span className={`inline-flex items-center gap-1 whitespace-nowrap border-[1.5px] font-semibold ${style.cls} ${size === "sm" ? "px-1.5 text-xs" : "px-2 py-0.5 text-sm"}`}>
      <Icon className={size === "sm" ? "h-3 w-3" : "h-3.5 w-3.5"} aria-hidden />
      {risk}
    </span>
  );
}

export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "accent" | "success" | "warning" | "danger" }) {
  const cls = {
    neutral: "text-muted",
    accent: "text-text",
    success: "text-success",
    warning: "text-warning",
    danger: "text-danger",
  }[tone];
  return <span className={`inline-flex items-center gap-1.5 text-sm font-semibold ${cls}`}>{children}</span>;
}

// ---------- numbers ----------

export function KpiCard({ label, value, decimals = 0, suffix = "", hint }: {
  label: string;
  value: number | null | undefined;
  decimals?: number;
  suffix?: string;
  hint?: ReactNode;
  icon?: ReactNode;
  delay?: number;
  tone?: "accent" | "success" | "warning" | "danger";
}) {
  const reduce = useReducedMotion();
  return (
    <div className="card pt-2">
      <div className="text-sm font-medium">{label}</div>
      <div className="num mt-1 font-display text-6xl font-semibold leading-none tracking-tight">
        {value === null || value === undefined ? <span className="text-muted">–</span>
          : reduce ? value.toFixed(decimals)
          : <RollCount to={value} decimals={decimals} mode="roll" duration={1.1} trigger="load" />}
        {value !== null && value !== undefined && suffix && <span className="ml-1 text-2xl font-medium text-muted">{suffix.trim()}</span>}
      </div>
      {hint && <div className="mt-2 text-sm leading-snug text-muted">{hint}</div>}
    </div>
  );
}

// ---------- feedback ----------

export function ProgressBar({ value, label }: { value: number; label?: string }) {
  const percent = Math.max(2, Math.round(value * 100));
  return (
    <div>
      {label && (
        <div className="mb-2 flex justify-between text-sm">
          <span className="truncate pr-2">{label}</span>
          <span className="num font-semibold">{Math.round(value * 100)}%</span>
        </div>
      )}
      <div className="relative h-5 border-b-[5px] border-border" role="progressbar" aria-valuenow={Math.round(value * 100)} aria-valuemin={0} aria-valuemax={100} aria-label={label}>
        <div className="absolute -bottom-[5px] left-0 h-[5px] bg-text transition-[width] duration-500" style={{ width: `${percent}%` }} />
        <div className="absolute bottom-0 flex h-[15px] w-9 items-center justify-around bg-accent px-1 transition-[left] duration-500" style={{ left: `calc(${percent}% - ${(percent / 100) * 36}px)` }}>
          {[0, 1, 2].map((i) => <i key={i} className="h-1.5 w-1.5 bg-onaccent" />)}
        </div>
      </div>
    </div>
  );
}

export function ErrorBanner({ message, onClose }: { message: string | null; onClose?: () => void }) {
  if (!message) return null;
  return (
    <div role="alert" className="mb-5 flex items-start gap-3 border-2 border-danger px-4 py-3 text-base font-medium text-danger">
      <OctagonAlert className="mt-0.5 h-5 w-5 shrink-0" />
      <span className="flex-1">{message}</span>
      {onClose && (
        <button type="button" onClick={onClose} className="focus-ring p-0.5" aria-label="Dismiss">
          <X className="h-5 w-5" />
        </button>
      )}
    </div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton ${className}`} role="status" aria-label="Loading" />;
}

export function EmptyState({ title, text, action }: { icon?: ReactNode; title: string; text: string; action?: ReactNode }) {
  return (
    <div className="border-y border-border py-10">
      <p className="font-display text-3xl font-semibold">{title}</p>
      <p className="mb-5 mt-1 max-w-lg text-muted">{text}</p>
      {action}
    </div>
  );
}

export function Stat({ label, value, tone }: { label: string; value: ReactNode; tone?: "success" | "danger" | "warning" | "accent" }) {
  const toneCls = tone ? { success: "text-success", danger: "text-danger", warning: "text-warning", accent: "text-accent" }[tone] : "";
  return (
    <div className="border-t border-border pt-1">
      <div className="text-xs text-muted">{label}</div>
      <div className={`num font-display text-2xl font-semibold leading-tight ${toneCls}`}>{value}</div>
    </div>
  );
}

export const fmt = (value: number | null | undefined, digits = 2) =>
  value === null || value === undefined || Number.isNaN(value) ? "n/a" : value.toFixed(digits);
