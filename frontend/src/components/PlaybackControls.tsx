import { FastForward, Pause, Play, RotateCcw, SkipBack, SkipForward } from "lucide-react";
import type { usePlayback } from "../lib/usePlayback";

type Playback = ReturnType<typeof usePlayback>;

const SPEEDS = [
  { label: "0.5x", delay: 260 },
  { label: "1x", delay: 130 },
  { label: "2x", delay: 60 },
  { label: "4x", delay: 20 },
];

export function PlaybackControls({ playback, total, label = "Step" }: { playback: Playback; total: number; label?: string }) {
  const { step, playing, play, pause, reset, forward, back, finish, delay, setDelay } = playback;
  const iconButton =
    "focus-ring grid h-8 w-8 place-items-center rounded-lg border border-border text-text transition-colors hover:border-text";
  return (
    <div className="flex flex-wrap items-center gap-2">
      <button className={iconButton} onClick={reset} aria-label="Reset"><RotateCcw className="h-4 w-4" /></button>
      <button className={iconButton} onClick={back} aria-label="Step back"><SkipBack className="h-4 w-4" /></button>
      <button
        className="focus-ring grid h-9 w-12 place-items-center bg-accent text-onaccent transition-opacity hover:opacity-80"
        onClick={playing ? pause : play}
        aria-label={playing ? "Pause" : "Play"}
      >
        {playing ? <Pause className="h-4 w-4" /> : <Play className="ml-0.5 h-4 w-4" />}
      </button>
      <button className={iconButton} onClick={forward} aria-label="Step forward"><SkipForward className="h-4 w-4" /></button>
      <button className={iconButton} onClick={finish} aria-label="Skip to end"><FastForward className="h-4 w-4" /></button>
      <div className="ml-1 inline-flex gap-1" role="group" aria-label="Speed">
        {SPEEDS.map((speed) => (
          <button
            key={speed.label}
            onClick={() => setDelay(speed.delay)}
            className={`focus-ring rounded-md px-2 py-1 text-xs font-medium transition-colors ${
              delay === speed.delay ? "bg-text text-bg" : "text-muted hover:text-text"}`}
          >
            {speed.label}
          </button>
        ))}
      </div>
      <span className="num ml-auto text-xs text-muted">{label} {Math.min(step, total)} / {total}</span>
    </div>
  );
}
