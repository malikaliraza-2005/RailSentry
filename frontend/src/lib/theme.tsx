import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";

export type ThemeChoice = "dark" | "light" | "system";
type Resolved = "dark" | "light";

const STORAGE_KEY = "railway-theme";

function readChoice(): ThemeChoice {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "dark" || saved === "light" || saved === "system") return saved;
  } catch {
    /* storage can be blocked; fall back to dark */
  }
  return "dark";
}

function resolve(choice: ThemeChoice): Resolved {
  if (choice !== "system") return choice;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

const ThemeContext = createContext<{
  choice: ThemeChoice;
  theme: Resolved;
  setChoice: (choice: ThemeChoice) => void;
}>({ choice: "dark", theme: "dark", setChoice: () => {} });

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [choice, setChoiceState] = useState<ThemeChoice>(readChoice);
  const [theme, setTheme] = useState<Resolved>(() => resolve(readChoice()));

  const apply = useCallback((next: Resolved) => {
    const root = document.documentElement;
    root.classList.add("theme-transition");
    root.setAttribute("data-theme", next);
    setTheme(next);
    window.setTimeout(() => root.classList.remove("theme-transition"), 250);
  }, []);

  const setChoice = useCallback(
    (next: ThemeChoice) => {
      setChoiceState(next);
      try {
        localStorage.setItem(STORAGE_KEY, next);
      } catch {
        /* ignore */
      }
      apply(resolve(next));
    },
    [apply],
  );

  // Follow the operating system when "system" is chosen.
  useEffect(() => {
    if (choice !== "system") return;
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => apply(resolve("system"));
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, [choice, apply]);

  return <ThemeContext.Provider value={{ choice, theme, setChoice }}>{children}</ThemeContext.Provider>;
}

export const useTheme = () => useContext(ThemeContext);

/** Read the current colour tokens (for charts and SVG), refreshed on theme change. */
export function useColors() {
  const { theme } = useTheme();
  const [colors, setColors] = useState(() => readColors());
  useEffect(() => {
    // wait one frame so the new data-theme attribute is applied
    const id = requestAnimationFrame(() => setColors(readColors()));
    return () => cancelAnimationFrame(id);
  }, [theme]);
  return colors;
}

function readColors() {
  const style = getComputedStyle(document.documentElement);
  const get = (name: string) => `rgb(${style.getPropertyValue(`--${name}`).trim().split(" ").join(",")})`;
  const alpha = (name: string, a: number) =>
    `rgba(${style.getPropertyValue(`--${name}`).trim().split(" ").join(",")},${a})`;
  return {
    bg: get("bg"),
    surface: get("surface"),
    surface2: get("surface-2"),
    border: get("border"),
    text: get("text"),
    muted: get("text-muted"),
    accent: get("accent"),
    success: get("success"),
    warning: get("warning"),
    danger: get("danger"),
    path: get("path"),
    alpha,
    series: [get("accent"), get("path"), get("warning"), get("success"), get("danger"), "rgb(167,139,250)"],
  };
}

export type Colors = ReturnType<typeof readColors>;
