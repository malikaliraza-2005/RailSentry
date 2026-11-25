import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, type JobEvent, type JobState } from "./api";

/**
 * Start a long-running backend job and poll it until it finishes.
 * Events (e.g. training epochs) are accumulated as they arrive.
 * If a job of the same kind is already running when the page opens,
 * the hook re-attaches to it.
 */
export function useJob<R = unknown>(kind: string, onDone?: (result: R) => void) {
  const [job, setJob] = useState<JobState<R> | null>(null);
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);
  const eventCount = useRef(0);
  const doneRef = useRef(onDone);
  doneRef.current = onDone;

  const stop = () => {
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = null;
  };

  const poll = useCallback((id: string) => {
    const tick = async () => {
      try {
        const state = await api.get<JobState<R>>(`/api/jobs/${id}?since=${eventCount.current}`);
        if (state.events.length) {
          eventCount.current += state.events.length;
          setEvents((previous) => [...previous, ...state.events]);
        }
        setJob(state);
        if (state.status === "done") {
          if (state.result && doneRef.current) doneRef.current(state.result);
          return;
        }
        if (state.status === "error") {
          setError(state.error || "The task failed.");
          return;
        }
      } catch (err) {
        setError(err instanceof ApiError ? err.message : String(err));
        return;
      }
      timer.current = window.setTimeout(tick, 600);
    };
    tick();
  }, []);

  const start = useCallback(
    async (body?: unknown) => {
      stop();
      setError(null);
      setEvents([]);
      eventCount.current = 0;
      try {
        const state = await api.post<JobState<R>>(`/api/jobs/${kind}`, body);
        setJob(state);
        poll(state.id);
      } catch (err) {
        setError(err instanceof ApiError ? err.message : String(err));
      }
    },
    [kind, poll],
  );

  // Re-attach to a running job of this kind (e.g. after switching pages).
  useEffect(() => {
    let cancelled = false;
    api
      .get<{ job: JobState<R> | null }>("/api/jobs/current")
      .then(({ job: current }) => {
        if (cancelled || !current || current.kind !== kind) return;
        if (current.status === "running" || current.status === "queued") {
          eventCount.current = 0;
          setEvents([]);
          setJob(current);
          poll(current.id);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
      stop();
    };
  }, [kind, poll]);

  const running = job?.status === "running" || job?.status === "queued";
  return { job, events, error, running, start, clearError: () => setError(null) };
}
