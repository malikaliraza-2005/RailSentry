import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { api, type Overview } from "./api";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyResult = any;

interface Store {
  overview: Overview | null;
  overviewError: string | null;
  refreshOverview: (overwrite?: boolean) => Promise<void>;
  results: Record<string, AnyResult>;
  setResult: (module: string, value: AnyResult) => void;
}

const StoreContext = createContext<Store>({
  overview: null,
  overviewError: null,
  refreshOverview: async () => {},
  results: {},
  setResult: () => {},
});

/** Keeps the latest result of each module, so pages keep their data when you switch. */
export function StoreProvider({ children }: { children: ReactNode }) {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [overviewError, setOverviewError] = useState<string | null>(null);
  const [results, setResults] = useState<Record<string, AnyResult>>({});

  const refreshOverview = useCallback(async (overwrite = false) => {
    try {
      const data = await api.get<Overview>("/api/overview");
      setOverview(data);
      setOverviewError(null);
      // Results saved by earlier runs are loaded once, unless newer ones exist.
      setResults((current) => {
        const next = { ...current };
        for (const key of ["module2", "module3"]) {
          if ((overwrite || !next[key]) && data.results[key]) next[key] = data.results[key];
        }
        return next;
      });
    } catch (error) {
      setOverviewError(error instanceof Error ? error.message : String(error));
    }
  }, []);

  useEffect(() => {
    refreshOverview();
  }, [refreshOverview]);

  const setResult = useCallback((module: string, value: AnyResult) => {
    setResults((current) => ({ ...current, [module]: value }));
  }, []);

  return (
    <StoreContext.Provider value={{ overview, overviewError, refreshOverview, results, setResult }}>
      {children}
    </StoreContext.Provider>
  );
}

export const useStore = () => useContext(StoreContext);
