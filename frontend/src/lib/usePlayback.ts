import { useCallback, useEffect, useRef, useState } from "react";

/**
 * A step counter for animations: play / pause / step / reset at a chosen speed.
 * `step` runs from 0 to `length`.
 */
export function usePlayback(length: number, initialDelay = 120) {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [delay, setDelay] = useState(initialDelay);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    if (!playing) return;
    if (step >= length) {
      setPlaying(false);
      return;
    }
    timer.current = window.setTimeout(() => setStep((s) => Math.min(length, s + 1)), delay);
    return () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    };
  }, [playing, step, length, delay]);

  const play = useCallback(() => {
    setStep((s) => (s >= length ? 0 : s));
    setPlaying(true);
  }, [length]);
  const pause = useCallback(() => setPlaying(false), []);
  const reset = useCallback(() => {
    setPlaying(false);
    setStep(0);
  }, []);
  const forward = useCallback(() => {
    setPlaying(false);
    setStep((s) => Math.min(length, s + 1));
  }, [length]);
  const back = useCallback(() => {
    setPlaying(false);
    setStep((s) => Math.max(0, s - 1));
  }, []);
  const finish = useCallback(() => {
    setPlaying(false);
    setStep(length);
  }, [length]);

  return { step, setStep, playing, play, pause, reset, forward, back, finish, delay, setDelay, done: step >= length };
}
