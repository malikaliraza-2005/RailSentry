"""
FastAPI backend for the RailSentry web interface.

It wraps the four project modules without changing their logic.

Run it with:
    C:\\venvs\\ailab\\Scripts\\python.exe -m uvicorn api:app --port 8000
(from this folder), then start the React frontend in ./frontend.
"""

from __future__ import annotations

import math
import sys
import threading
import time
import traceback
import uuid
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Charts are saved to files only; no windows open on the server.
import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message=".*non-interactive.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import json

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Capture everything the modules print, for the Logs page
# --------------------------------------------------------------------------

class LogBuffer:
    """Keeps the last lines written to stdout/stderr, and still prints them."""

    def __init__(self, stream, buffer: deque, counter: List[int], lock: threading.Lock):
        self.stream = stream
        self.buffer = buffer
        self.counter = counter
        self.lock = lock
        self.partial = ""

    def write(self, text: str) -> int:
        try:
            self.stream.write(text)
        except Exception:
            pass
        with self.lock:
            self.partial += text
            while "\n" in self.partial:
                line, self.partial = self.partial.split("\n", 1)
                self.counter[0] += 1
                self.buffer.append({"id": self.counter[0], "time": time.time(), "text": line})
        return len(text)

    def flush(self) -> None:
        try:
            self.stream.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return False


LOG_LINES: deque = deque(maxlen=3000)
LOG_COUNTER = [0]
LOG_LOCK = threading.Lock()
sys.stdout = LogBuffer(sys.stdout, LOG_LINES, LOG_COUNTER, LOG_LOCK)
sys.stderr = LogBuffer(sys.stderr, LOG_LINES, LOG_COUNTER, LOG_LOCK)


import module1_agent_search as module1  # noqa: E402
import module2_data_ml as module2  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"
STATE_FILE = OUTPUT_DIR / "ui_state.json"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Only one module task at a time: matplotlib and TensorFlow are not safe to
# run in parallel, and tasks overwrite the same model files.
RUN_LOCK = threading.Lock()


def module3():
    """Import Module 3 (TensorFlow) only when it is first needed."""
    import module3_neural_network
    return module3_neural_network


def module4():
    import module4_integrated_system
    return module4_integrated_system


# --------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------

def to_json(value: Any) -> Any:
    """Convert NumPy / pandas values (and NaN) into plain JSON values."""
    if isinstance(value, dict):
        return {str(key): to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return to_json(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return to_json(value.tolist())
    if isinstance(value, np.ndarray):
        return to_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if math.isnan(number) or math.isinf(number) else number
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(key: str, value: Any) -> None:
    state = load_state()
    state[key] = to_json(value)
    state[key + "_updated"] = time.time()
    STATE_FILE.write_text(json.dumps(state), encoding="utf-8")


def busy_guard(function: Callable, timeout: float = 3.0):
    """Run a quick task under the module lock, or report that another task is running."""
    if not RUN_LOCK.acquire(timeout=timeout):
        raise HTTPException(status_code=409, detail="Another task is running. Try again when it finishes.")
    try:
        return function()
    finally:
        RUN_LOCK.release()


# --------------------------------------------------------------------------
# Background jobs (long tasks report progress and are polled by the UI)
# --------------------------------------------------------------------------

class Job:
    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.status = "queued"
        self.step = ""
        self.progress = 0.0
        self.events: List[dict] = []
        self.result: Any = None
        self.error: Optional[str] = None
        self.started = time.time()
        self.finished: Optional[float] = None

    def report(self, step: str = None, progress: float = None, event: dict = None):
        if step is not None:
            self.step = step
        if progress is not None:
            self.progress = float(progress)
        if event is not None:
            self.events.append(event)

    def as_dict(self, since_event: int = 0) -> dict:
        return to_json({
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "step": self.step,
            "progress": self.progress,
            "events": self.events[since_event:],
            "event_count": len(self.events),
            "result": self.result if self.status == "done" else None,
            "error": self.error,
            "started": self.started,
            "finished": self.finished,
        })


JOBS: Dict[str, Job] = {}
CURRENT_JOB: List[Optional[str]] = [None]


def start_job(kind: str, work: Callable[[Job], Any]) -> Job:
    if not RUN_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Another task is already running.")

    job = Job(kind)
    JOBS[job.id] = job
    CURRENT_JOB[0] = job.id

    def runner():
        job.status = "running"
        try:
            job.result = to_json(work(job))
            job.status = "done"
            job.progress = 1.0
        except Exception as error:  # report any failure to the UI
            job.status = "error"
            job.error = f"{type(error).__name__}: {error}"
            print(traceback.format_exc())
        finally:
            job.finished = time.time()
            RUN_LOCK.release()

    threading.Thread(target=runner, daemon=True).start()
    return job


# --------------------------------------------------------------------------
# Result summaries for each module
# --------------------------------------------------------------------------

def grid_to_list(grid: np.ndarray) -> List[List[int]]:
    return [[int(value) for value in row] for row in grid]


def search_summary(grid: np.ndarray, start, goal) -> dict:
    results = module1.run_all_searches(grid, tuple(start), tuple(goal))
    summary = {}
    for name, result in results.items():
        summary[name] = {
            "path": result["path"],
            "explored": result["explored"],
            "path_length": result["path_length"],
            "cost": result["cost"],
            "nodes_expanded": result["nodes_expanded"],
            "time_ms": result["time"] * 1000,
        }
    valid = {name: r for name, r in summary.items() if r["path"] is not None}
    best = min(valid, key=lambda n: (valid[n]["cost"], valid[n]["path_length"],
                                      valid[n]["nodes_expanded"])) if valid else None
    return {"results": summary, "best": best}


def layouts_payload() -> dict:
    return {
        name: {"grid": grid_to_list(layout["grid"]),
               "start": layout["start"], "goal": layout["goal"]}
        for name, layout in module1.create_layouts().items()
    }


def module1_summary() -> dict:
    module1.run_module1()
    return {name: {**layout, **search_summary(np.array(layout["grid"]), layout["start"], layout["goal"])}
            for name, layout in layouts_payload().items()}


def five_number(series: pd.Series) -> Optional[dict]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = values[(values >= low_fence) & (values <= high_fence)]
    return {
        "min": float(inside.min()), "q1": float(q1), "median": float(median),
        "q3": float(q3), "max": float(inside.max()),
        "outliers": sorted(values[(values < low_fence) | (values > high_fence)].tolist()),
    }


def module2_summary(result: dict) -> dict:
    data: pd.DataFrame = result["data"]
    raw: pd.DataFrame = result["raw_data"]
    kmeans, scaler = result["kmeans"], result["location_scaler"]

    response = data["response_time_min"].dropna()
    counts, edges = np.histogram(response, bins=20)
    histogram = [{"from": float(edges[i]), "to": float(edges[i + 1]), "count": int(counts[i])}
                 for i in range(len(counts))]

    corr_columns = module2.NUMERIC_COLUMNS + ["response_time_min"]
    correlation = data[corr_columns].corr(numeric_only=True)

    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    classifier, regressor = result["classifier"], result["regressor"]

    return {
        "records_raw": len(raw),
        "records_clean": len(data),
        "cleaning_report": result["cleaning_report"],
        "missing_after": data.isna().sum().to_dict(),
        "risk_counts": data["risk_level"].value_counts().reindex(module2.RISK_ORDER, fill_value=0).to_dict(),
        "incident_types": data["incident_type"].value_counts().to_dict(),
        "histogram": histogram,
        "correlation": {"columns": corr_columns, "matrix": correlation.round(3).values.tolist()},
        "boxplot": {column: five_number(raw[column]) for column in module2.NUMERIC_COLUMNS},
        "elbow": result["elbow"],
        "clusters": {
            "points": data[["grid_x", "grid_y", "location_cluster", "risk_level"]].to_dict(orient="records"),
            "centroids": [{"grid_x": float(x), "grid_y": float(y), "cluster": i}
                          for i, (x, y) in enumerate(centroids)],
        },
        "new_location_cluster": result["new_location_cluster"],
        "knn_classifier": {
            "tuning": classifier["tuning_table"],
            "best_k": classifier["best_k"],
            "best_metric": classifier["best_metric"],
            "test_accuracy": classifier["test_accuracy"],
            "test_macro_f1": classifier["test_macro_f1"],
        },
        "knn_regressor": {
            "best_k": regressor["best_k"],
            "best_metric": regressor["best_metric"],
            "mae": regressor["mae"], "mse": regressor["mse"], "rmse": regressor["rmse"],
        },
    }


def module3_summary(result: dict) -> dict:
    return {
        "best_ann": result["best_ann_name"],
        "comparison": result["comparison"],
        "validation_metrics": result["validation_metrics"],
        "histories": result["histories"],
        "confusion_matrices": result["confusion_matrices"],
        "roc": result["roc"],
        "wrong_predictions": result["wrong_predictions"],
        "knn": {"k": result["knn_k"], "metric": result["knn_metric"]},
        "regressor": {"mae": result["knn_mae"], "mse": result["knn_mse"]},
        "split_sizes": result["split_sizes"],
    }


def routing_summary(incidents: pd.DataFrame = None) -> dict:
    m4 = module4()
    result = m4.compare_routing(incidents)
    return {
        "grid": grid_to_list(result["grid"]),
        "start": result["start"], "goal": result["goal"],
        "actual_map": grid_to_list(result["actual_map"]),
        "predicted_map": grid_to_list(result["predicted_map"]),
        "normal_path": result["plain"]["path"],
        "risk_aware_path": result["risk_aware"]["path"],
        "table": result["table"],
    }


def replan_summary(step: int = 6) -> dict:
    result = module4().demonstrate_replanning(step)
    if not result:
        return {}
    return {**result, "grid": grid_to_list(result["grid"])}


def experiments_summary(results: pd.DataFrame) -> dict:
    summary = results.groupby(["Layout", "Layout name", "Mode"], sort=False).mean(numeric_only=True)
    summary = summary.drop(columns=["Run"]).reset_index()
    return {"runs": results, "summary": summary}


def module4_summary() -> dict:
    m4 = module4()
    incidents = m4.load_incidents_with_predictions()
    targets = m4.demonstrate_multi_target(incidents)
    return {
        "routing": routing_summary(incidents),
        "targets": targets,
        "replan": replan_summary(),
        "experiments": experiments_summary(m4.run_experiments(incidents)),
        "settings": {
            "predicted_risk_penalty": m4.PREDICTED_RISK_PENALTY,
            "actual_risk_delay": m4.ACTUAL_RISK_DELAY,
            "incidents_per_run": m4.INCIDENTS_PER_RUN,
            "runs_per_layout": m4.RUNS_PER_LAYOUT,
        },
    }


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

app = FastAPI(title="RailSentry API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    grid: List[List[int]]
    start: List[int] = Field(min_length=2, max_length=2)
    goal: List[int] = Field(min_length=2, max_length=2)


class LocationRequest(BaseModel):
    grid_x: float = Field(ge=0, le=14)
    grid_y: float = Field(ge=0, le=14)


class CsvRequest(BaseModel):
    text: str = Field(max_length=1_000_000)


class AnalyzeRequest(BaseModel):
    rows: List[Dict[str, Any]]
    grid: List[List[int]]
    start: List[int] = Field(min_length=2, max_length=2)


class IncidentRequest(BaseModel):
    grid_x: float
    grid_y: float
    incident_type: str
    crowd_density_pct: float
    hazard_severity_1_5: float
    equipment_health_pct: float
    obstruction_level_0_3: float
    hour_of_day: int


@app.get("/api/health")
def health():
    return {"status": "ok", "busy": RUN_LOCK.locked(), "current_job": CURRENT_JOB[0]}


@app.get("/api/overview")
def overview():
    data = module2.load_prepared_data()
    model_files = {
        "KNN risk classifier (Module 2)": "knn_risk_classifier.joblib",
        "KNN response-time regressor (Module 2)": "knn_response_time_regressor.joblib",
        "K-Means location clusters": "location_kmeans.joblib",
        "ANN architecture 1": "ann_architecture_1.keras",
        "ANN architecture 2": "ann_architecture_2.keras",
        "Best risk model (ANN)": "best_risk_model.keras",
        "Module 3 preprocessor": "module3_preprocessor.joblib",
        "Module 3 KNN regressor": "module3_knn_regressor.joblib",
    }
    return to_json({
        "dataset": {
            "records": len(data),
            "risk_counts": data["risk_level"].value_counts().reindex(module2.RISK_ORDER, fill_value=0).to_dict(),
            "incident_types": data["incident_type"].value_counts().to_dict(),
            "average_response_time": data["response_time_min"].mean(),
        },
        "models": {label: (MODEL_DIR / name).exists() for label, name in model_files.items()},
        "results": load_state(),
        "busy": RUN_LOCK.locked(),
        "current_job": CURRENT_JOB[0],
    })


@app.get("/api/dataset")
def dataset():
    """Raw records, with the fields the cleaning step changed."""
    raw = pd.read_csv(module2.find_dataset())
    report: dict = {}
    module2.handle_outliers(module2.handle_missing_values(raw), report)
    cleaned_fields: Dict[str, List[str]] = {}
    for column, info in report.items():
        for incident_id in info["incident_ids"]:
            cleaned_fields.setdefault(incident_id, []).append(column)
    records = raw.to_dict(orient="records")
    for record in records:
        record["cleaned_fields"] = cleaned_fields.get(record["incident_id"], [])
        record["missing_fields"] = [key for key, value in record.items()
                                    if isinstance(value, float) and math.isnan(value)]
    return to_json({"records": records, "report": report})


@app.get("/api/layouts")
def layouts():
    return to_json(layouts_payload())


@app.post("/api/search")
def search(request: SearchRequest):
    grid = np.array(request.grid, dtype=np.int64)
    if grid.ndim != 2 or grid.shape[0] < 5 or grid.shape[1] < 5:
        raise HTTPException(400, "The grid must be a 2-D list of at least 5x5 cells.")
    for name, cell in (("start", request.start), ("goal", request.goal)):
        if not (0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1]):
            raise HTTPException(400, f"The {name} cell is outside the grid.")
        if grid[cell[0], cell[1]] == -1:
            raise HTTPException(400, f"The {name} cell cannot be blocked.")
    return to_json(search_summary(grid, request.start, request.goal))


@app.get("/api/agent/scenarios")
def agent_scenarios():
    station_map = module1.create_weighted_grid(15, 15)
    output = []
    for scenario in module1.build_agent_scenarios():
        agent = module1.RailwayResponseAgent(scenario["start"], scenario["target"], station_map)
        result = agent.run_episode(scenario["grid"])
        output.append({
            "name": scenario["name"],
            "description": scenario["description"],
            "grid": grid_to_list(scenario["grid"]),
            "start": scenario["start"],
            "target": scenario["target"],
            "trace": agent.trace,
            "path_taken": agent.path_taken,
            "result": result,
        })
    return to_json({"scenarios": output,
                    "performance_formula": "100 x success - route cost - 5 x re-plans - waits"})


@app.post("/api/cluster")
def cluster(request: LocationRequest):
    try:
        kmeans = joblib.load(MODEL_DIR / "location_kmeans.joblib")
        scaler = joblib.load(MODEL_DIR / "location_scaler.joblib")
    except FileNotFoundError:
        raise HTTPException(404, "Run Module 2 first to create the location clusters.")
    return {"cluster": module2.predict_new_cluster(request.grid_x, request.grid_y, kmeans, scaler)}


@app.post("/api/predict")
def predict(request: IncidentRequest):
    def work():
        return module3().predict_incident(request.model_dump())
    try:
        return to_json(busy_guard(work, timeout=10))
    except ValueError as error:
        raise HTTPException(400, str(error))
    except FileNotFoundError as error:
        raise HTTPException(404, str(error))


@app.post("/api/user-data/parse")
def parse_user_csv(request: CsvRequest):
    """Read an uploaded CSV and report every problem, without running the models."""
    import user_data
    try:
        parsed = user_data.parse_csv(request.text)
    except ValueError as error:
        raise HTTPException(400, str(error))
    _, errors, warnings = user_data.validate_rows(parsed["rows"])
    return to_json({**parsed, "errors": errors, "warnings": warnings})


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest):
    """Predict risk, response time, cluster, priority and routes for user incidents."""
    import user_data
    try:
        return to_json(busy_guard(
            lambda: user_data.analyse(request.rows, request.grid, request.start), timeout=10))
    except FileNotFoundError as error:
        raise HTTPException(404, str(error))
    except ValueError as error:
        raise HTTPException(400, str(error))


@app.get("/api/module4/routing")
def module4_routing():
    try:
        return to_json(busy_guard(routing_summary, timeout=10))
    except FileNotFoundError as error:
        raise HTTPException(404, str(error))


@app.get("/api/module4/replan")
def module4_replan(step: int = 6):
    return to_json(busy_guard(lambda: replan_summary(step), timeout=10))


@app.get("/api/module4/targets")
def module4_targets():
    def work():
        m4 = module4()
        return m4.demonstrate_multi_target(m4.load_incidents_with_predictions())
    try:
        return to_json(busy_guard(work, timeout=10))
    except FileNotFoundError as error:
        raise HTTPException(404, str(error))


# ---- long-running jobs ----

@app.post("/api/jobs/module1")
def job_module1():
    def work(job: Job):
        job.report("Running agent demonstrations and searches", 0.1)
        result = module1_summary()
        save_state("module1", {name: {k: v for k, v in layout.items() if k != "grid"}
                               for name, layout in result.items()})
        return result
    return start_job("module1", work).as_dict()


@app.post("/api/jobs/module2")
def job_module2(request: LocationRequest = None):
    location = (request.grid_x, request.grid_y) if request else (7, 7)

    def work(job: Job):
        job.report("Cleaning data, clustering and training KNN", 0.1)
        summary = module2_summary(module2.run_module2(new_location=location))
        save_state("module2", summary)
        return summary
    return start_job("module2", work).as_dict()


def train_module3(job: Job, base: float = 0.0, span: float = 1.0) -> dict:
    max_epochs = 50 * 2

    def on_epoch(event: dict):
        done = sum(1 for e in job.events if "epoch" in e) + 1
        job.report(f"Training {event['model']} - epoch {event['epoch']}",
                   base + span * min(0.9, done / max_epochs), event)

    job.report("Preparing data", base)
    result = module3().run_module3(ask_to_predict=False, epoch_callback=on_epoch)
    job.report("Evaluating models", base + span * 0.95)
    summary = module3_summary(result)
    save_state("module3", summary)
    return summary


@app.post("/api/jobs/module3")
def job_module3():
    return start_job("module3", train_module3).as_dict()


@app.post("/api/jobs/module4")
def job_module4():
    def work(job: Job):
        job.report("Predicting incident risk and running 30 experiments", 0.1)
        summary = module4_summary()
        save_state("module4", {
            "summary": summary["experiments"]["summary"],
            "routing_table": summary["routing"]["table"],
        })
        return summary
    try:
        return start_job("module4", work).as_dict()
    except FileNotFoundError as error:
        raise HTTPException(404, str(error))


@app.post("/api/jobs/full_demo")
def job_full_demo():
    def work(job: Job):
        output = {}
        job.report("Module 1 - agent and search", 0.02, {"stage": "module1", "status": "running"})
        output["module1"] = module1_summary()
        job.report(None, 0.1, {"stage": "module1", "status": "done"})

        job.report("Module 2 - data and machine learning", 0.12, {"stage": "module2", "status": "running"})
        output["module2"] = module2_summary(module2.run_module2(new_location=(7, 7)))
        save_state("module2", output["module2"])
        job.report(None, 0.25, {"stage": "module2", "status": "done"})

        job.events.append({"stage": "module3", "status": "running"})
        output["module3"] = train_module3(job, base=0.25, span=0.6)
        job.report(None, 0.85, {"stage": "module3", "status": "done"})

        job.report("Module 4 - integrated smart system", 0.87, {"stage": "module4", "status": "running"})
        output["module4"] = module4_summary()
        save_state("module4", {"summary": output["module4"]["experiments"]["summary"],
                               "routing_table": output["module4"]["routing"]["table"]})
        job.report("Complete", 1.0, {"stage": "module4", "status": "done"})
        return {"modules": list(output.keys())}
    return start_job("full_demo", work).as_dict()


@app.get("/api/jobs/current")
def current_job():
    job_id = CURRENT_JOB[0]
    if job_id is None or job_id not in JOBS:
        return {"job": None}
    return {"job": JOBS[job_id].as_dict()}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, since: int = 0):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    return job.as_dict(since_event=since)


@app.get("/api/logs")
def logs(after: int = 0, limit: int = 500):
    with LOG_LOCK:
        lines = [line for line in LOG_LINES if line["id"] > after]
    return {"lines": lines[-limit:], "last_id": LOG_COUNTER[0]}
