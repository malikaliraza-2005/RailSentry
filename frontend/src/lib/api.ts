// Small typed client for the FastAPI backend (proxied at /api by Vite).

export type Cell = [number, number];
export type GridData = number[][];
export type RiskName = "Low" | "Medium" | "High";
export const RISK_NAMES: RiskName[] = ["Low", "Medium", "High"];

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    });
  } catch {
    throw new ApiError(0, "Cannot reach the backend. Is the API server running on port 8000?");
  }
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail)) message = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      /* not JSON */
    }
    throw new ApiError(response.status, message);
  }
  return response.json() as Promise<T>;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) }),
};

// ---------- shared types (only the fields the UI uses) ----------

export interface SearchResult {
  path: Cell[] | null;
  explored: Cell[];
  path_length: number;
  cost: number;
  nodes_expanded: number;
  time_ms: number;
}

export interface Layout {
  grid: GridData;
  start: Cell;
  goal: Cell;
}

export interface JobState<R = unknown> {
  id: string;
  kind: string;
  status: "queued" | "running" | "done" | "error";
  step: string;
  progress: number;
  events: JobEvent[];
  event_count: number;
  result: R | null;
  error: string | null;
}

export interface JobEvent {
  model?: string;
  epoch?: number;
  loss?: number;
  accuracy?: number;
  val_loss?: number;
  val_accuracy?: number;
  stage?: string;
  status?: string;
}

export interface Overview {
  dataset: {
    records: number;
    risk_counts: Record<RiskName, number>;
    incident_types: Record<string, number>;
    average_response_time: number;
  };
  models: Record<string, boolean>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  results: Record<string, any>;
  busy: boolean;
  current_job: string | null;
}

export const cellKey = (cell: Cell) => `${cell[0]},${cell[1]}`;
