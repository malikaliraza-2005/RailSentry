import type { ReactNode } from "react";
import { useColors } from "../lib/theme";

/** Shared Recharts props so every chart follows the current theme. */
export function useChartTheme() {
  const colors = useColors();
  return {
    colors,
    grid: { stroke: colors.border, vertical: false },
    axis: {
      stroke: colors.border,
      tick: { fill: colors.muted, fontSize: 13, fontFamily: "Barlow, Arial, sans-serif" },
      tickLine: false,
    },
    tooltip: {
      cursor: { fill: colors.alpha("accent", 0.08), stroke: colors.border },
      contentStyle: {
        background: colors.surface,
        border: `2px solid ${colors.text}`,
        borderRadius: 0,
        color: colors.text,
        fontSize: 13,
      },
      labelStyle: { color: colors.muted, marginBottom: 4 },
      itemStyle: { color: colors.text },
    },
    legend: { wrapperStyle: { fontSize: 12, color: colors.muted } },
  };
}

/** A matrix heatmap (correlation, confusion matrix) drawn with CSS grid. */
export function Heatmap({ rows, columns, values, min = -1, max = 1, format = (v: number) => v.toFixed(2), diverging = true, rowTitle, columnTitle }: {
  rows: string[];
  columns: string[];
  values: number[][];
  min?: number;
  max?: number;
  format?: (value: number) => string;
  diverging?: boolean;
  rowTitle?: ReactNode;
  columnTitle?: ReactNode;
}) {
  const colors = useColors();
  const strength = (value: number) => (diverging
    ? Math.min(1, Math.abs(value) / Math.max(Math.abs(min), Math.abs(max)))
    : (value - min) / (max - min || 1));
  const colourFor = (value: number) => {
    if (diverging) {
      const t = Math.max(-1, Math.min(1, value / Math.max(Math.abs(min), Math.abs(max))));
      return t >= 0 ? colors.alpha("accent", 0.1 + 0.8 * t) : colors.alpha("danger", 0.1 + 0.8 * -t);
    }
    const t = (value - min) / (max - min || 1);
    return colors.alpha("accent", 0.06 + 0.85 * t);
  };
  return (
    <div className="overflow-x-auto">
      {columnTitle && <div className="mb-1 pl-28 text-center text-xs text-muted">{columnTitle}</div>}
      <div className="inline-grid min-w-full gap-1" style={{ gridTemplateColumns: `7rem repeat(${columns.length}, minmax(3rem, 1fr))` }}>
        <div className="flex items-end text-sm text-muted">{rowTitle}</div>
        {columns.map((column) => (
          <div key={column} className="truncate px-1 text-center text-sm text-muted" title={column}>{column}</div>
        ))}
        {rows.map((row, r) => (
          <div key={row} className="contents">
            <div className="truncate pr-2 text-right text-sm leading-9 text-muted" title={row}>{row}</div>
            {values[r].map((value, c) => (
              <div
                key={c}
                className="num grid h-9 place-items-center rounded-md text-sm font-semibold"
                style={{ background: colourFor(value), color: strength(value) > 0.5 ? colors.bg : colors.text }}
                title={`${row} / ${columns[c]}: ${format(value)}`}
              >
                {format(value)}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
