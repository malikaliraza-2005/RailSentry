import { motion, useReducedMotion } from "framer-motion";
import { memo, useId, useMemo } from "react";
import { cellKey, type Cell, type GridData } from "../lib/api";
import { useColors } from "../lib/theme";

export interface PathLayer {
  id: string;
  cells: Cell[] | null | undefined;
  color: string;
  dashed?: boolean;
  width?: number;
  duration?: number;
}

export interface Marker {
  cell: Cell;
  color: string;
  kind: "x" | "ring" | "dot";
  pulse?: boolean;
  title?: string;
}

interface Props {
  grid: GridData;
  start?: Cell | null;
  goal?: Cell | null;
  explored?: Cell[];
  exploredColor?: string;
  paths?: PathLayer[];
  hazards?: number[][] | null;
  vehicle?: Cell | null;
  markers?: Marker[];
  /** blocked cells the agent does not know about yet (drawn faintly) */
  unknownBlocked?: Set<string>;
  showCosts?: boolean;
  onCellClick?: (cell: Cell) => void;
  label: string;
  className?: string;
}

const SIZE = 40;
const centre = (cell: Cell) => ({ x: cell[1] * SIZE + SIZE / 2, y: cell[0] * SIZE + SIZE / 2 });

function StationGridInner({
  grid, start, goal, explored = [], exploredColor, paths = [], hazards, vehicle,
  markers = [], unknownBlocked, showCosts = false, onCellClick, label, className = "",
}: Props) {
  const colors = useColors();
  const reduce = useReducedMotion();
  const hatch = `hatch-${useId().replace(/[^a-zA-Z0-9]/g, "")}`;
  const rows = grid.length;
  const cols = grid[0]?.length ?? 0;
  const maxCost = useMemo(() => Math.max(2, ...grid.flat().filter((v) => v > 0)), [grid]);
  const exploredSet = useMemo(() => explored.map(cellKey), [explored]);

  return (
    <svg
      viewBox={`0 0 ${cols * SIZE} ${rows * SIZE}`}
      className={`h-auto w-full select-none ${className}`}
      role="img"
      aria-label={label}
    >
      <defs>
        <pattern id={hatch} width={8} height={8} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width={8} height={8} fill={colors.surface} />
          <rect width={3.2} height={8} fill={colors.text} />
        </pattern>
      </defs>
      {/* cells */}
      {grid.map((row, r) =>
        row.map((value, c) => {
          const key = `${r},${c}`;
          const blocked = value === -1;
          const hidden = blocked && unknownBlocked?.has(key);
          const shade = blocked ? 0 : ((value - 1) / (maxCost - 1)) * 0.22;
          return (
            <g key={key} onClick={onCellClick ? () => onCellClick([r, c]) : undefined}
               className={onCellClick ? "cursor-pointer" : undefined}>
              <rect
                x={c * SIZE} y={r * SIZE} width={SIZE} height={SIZE}
                fill={blocked && !hidden ? `url(#${hatch})` : colors.surface}
                stroke={hidden ? colors.muted : colors.alpha("border", 0.7)}
                strokeWidth={hidden ? 2 : 1}
                strokeDasharray={hidden ? "4 3" : undefined}
              />
              {!blocked && shade > 0 && !showCosts && (
                <circle cx={c * SIZE + SIZE / 2} cy={r * SIZE + SIZE / 2} r={2 + shade * 16} fill={colors.alpha("text-muted", 0.55)} />
              )}
              {showCosts && !blocked && (
                <text x={c * SIZE + SIZE / 2} y={r * SIZE + SIZE / 2 + 4} textAnchor="middle"
                      fontSize={15} fill={colors.muted} className="num" pointerEvents="none">
                  {value}
                </text>
              )}
              <title>{blocked ? `(${r}, ${c}) blocked` : `(${r}, ${c}) cost ${value}`}</title>
            </g>
          );
        }),
      )}

      {/* hazards (risk 1 = medium, 2 = high) */}
      {hazards?.map((row, r) =>
        row.map((level, c) =>
          level > 0 && grid[r][c] !== -1 ? (
            <path key={`h${r},${c}`}
                  d={level === 2
                    ? `M${c * SIZE + 3} ${r * SIZE + 3}h${SIZE - 6}v${SIZE - 6}h-${SIZE - 6}z`
                    : `M${c * SIZE + SIZE / 2} ${r * SIZE + 4}L${c * SIZE + SIZE - 4} ${r * SIZE + SIZE - 5}H${c * SIZE + 4}Z`}
                  fill={level === 2 ? colors.danger : colors.warning}
                  className="hazard-pulse" style={{ animationDelay: `${((r * 7 + c * 3) % 10) / 10}s` }}
                  pointerEvents="none" />
          ) : null,
        ),
      )}

      {/* explored cells, in the order the algorithm expanded them */}
      {exploredSet.map((key, index) => {
        const [r, c] = key.split(",").map(Number);
        return (
          <motion.rect
            key={`e${key}`}
            x={c * SIZE + 1} y={r * SIZE + 1} width={SIZE - 2} height={SIZE - 2}
            fill={exploredColor ?? colors.alpha("accent", 0.35)}
            initial={reduce ? false : { opacity: 0, scale: 0.3 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.18 }}
            style={{ transformBox: "fill-box", transformOrigin: "center" }}
            pointerEvents="none"
            data-index={index}
          />
        );
      })}

      {/* paths draw themselves */}
      {paths.map((layer) =>
        layer.cells && layer.cells.length > 1 ? (
          <motion.path
            key={`${layer.id}-${layer.cells.length}-${cellKey(layer.cells[layer.cells.length - 1])}`}
            d={layer.cells.map((cell, i) => `${i ? "L" : "M"}${centre(cell).x} ${centre(cell).y}`).join(" ")}
            fill="none"
            stroke={layer.color}
            strokeWidth={layer.width ?? 5}
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeDasharray={layer.dashed ? "2 10" : undefined}
            // Drawing a line on with pathLength replaces its dashes, so dashed routes fade in.
            // framer-motion's types reject pathLength on SVG paths; it works at runtime.
            initial={reduce ? false : ((layer.dashed ? { opacity: 0 } : { pathLength: 0, opacity: 0.4 }) as unknown as false)}
            animate={(layer.dashed ? { opacity: 1 } : { pathLength: 1, opacity: 1 }) as unknown as undefined}
            transition={{ duration: layer.duration ?? 1, ease: "easeInOut" }}
            pointerEvents="none"
          />
        ) : null,
      )}

      {/* markers: new obstructions, discovered walls, targets */}
      {markers.map((marker, i) => {
        const { x, y } = centre(marker.cell);
        return (
          <g key={`m${i}-${cellKey(marker.cell)}`} pointerEvents="none">
            {marker.pulse && <circle cx={x} cy={y} r={14} fill="none" stroke={marker.color} strokeWidth={3} className="ping-ring" />}
            {marker.kind === "x" && (
              <motion.g initial={reduce ? false : { scale: 0 }} animate={{ scale: 1 }}
                        style={{ transformBox: "fill-box", transformOrigin: "center" }}
                        transition={{ type: "spring", stiffness: 400, damping: 18 }}>
                <rect x={x - 16} y={y - 16} width={32} height={32} fill={marker.color} />
                <path d={`M${x - 7} ${y - 7} L${x + 7} ${y + 7} M${x + 7} ${y - 7} L${x - 7} ${y + 7}`}
                      stroke="white" strokeWidth={3.5} strokeLinecap="round" />
              </motion.g>
            )}
            {marker.kind === "ring" && <circle cx={x} cy={y} r={12} fill="none" stroke={marker.color} strokeWidth={3.5} />}
            {marker.kind === "dot" && <circle cx={x} cy={y} r={6} fill={marker.color} />}
            {marker.title && <title>{marker.title}</title>}
          </g>
        );
      })}

      {/* start and goal */}
      {start && (
        <g pointerEvents="none">
          <circle cx={centre(start).x} cy={centre(start).y} r={15} fill={colors.surface} stroke={colors.text} strokeWidth={4} />
          <text x={centre(start).x} y={centre(start).y + 6} textAnchor="middle" fontSize={17}
                fontWeight={700} fill={colors.text}>S</text>
        </g>
      )}
      {goal && (
        <g pointerEvents="none">
          <path
            d={starPath(centre(goal).x, centre(goal).y, 15, 7)}
            fill={colors.text} stroke={colors.surface} strokeWidth={2.5}
          />
        </g>
      )}

      {/* the response vehicle */}
      {vehicle && (
        <motion.g
          initial={false}
          animate={{ x: centre(vehicle).x, y: centre(vehicle).y }}
          transition={reduce ? { duration: 0 } : { type: "spring", stiffness: 500, damping: 50 }}
          pointerEvents="none"
        >
          <rect x={-16} y={-10} width={32} height={20} rx={3} fill={colors.path} stroke={colors.surface} strokeWidth={2.5} />
          <path d="M-11 -5h6v6h-6zM-3 -5h6v6h-6zM5 -5h6v6h-6z" fill={colors.surface} />
        </motion.g>
      )}
    </svg>
  );
}

function starPath(cx: number, cy: number, outer: number, inner: number) {
  const points: string[] = [];
  for (let i = 0; i < 10; i++) {
    const radius = i % 2 === 0 ? outer : inner;
    const angle = (Math.PI / 5) * i - Math.PI / 2;
    points.push(`${cx + radius * Math.cos(angle)},${cy + radius * Math.sin(angle)}`);
  }
  return `M${points.join("L")}Z`;
}

export const StationGrid = memo(StationGridInner);

export function GridLegend({ items }: { items: { color: string; label: string; shape?: "square" | "line" | "dashed" | "circle" | "triangle" }[] }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-sm text-muted">
      {items.map((item) => (
        <span key={item.label} className="inline-flex items-center gap-1.5">
          {item.shape === "line" || item.shape === "dashed" ? (
            <span className="inline-block h-0 w-5 border-t-[3px]" style={{ borderColor: item.color, borderStyle: item.shape === "dashed" ? "dotted" : "solid" }} />
          ) : (
            <span className={`inline-block h-3 w-3 ${item.shape === "circle" ? "rounded-full" : "rounded-[3px]"}`} style={{ background: item.color, clipPath: item.shape === "triangle" ? "polygon(50% 0, 100% 100%, 0 100%)" : undefined }} />
          )}
          {item.label}
        </span>
      ))}
    </div>
  );
}
