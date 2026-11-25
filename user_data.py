# Module description: Analyse incidents supplied by the user (form or CSV)

"""
Checks user-supplied incidents and runs them through the trained models:
    - risk level            (Module 3 ANN)
    - response time         (Module 3 KNN regressor)
    - location cluster      (Module 2 K-Means)
    - priority and routes   (Module 4, using Module 1's A*)

Nothing is retrained and nothing is saved. User data has no true risk
labels, so every risk value returned here is a prediction.
"""

import csv
import io
import math

import joblib
import numpy as np
import pandas as pd

import module2_data_ml as module2
import module4_integrated_system as module4


MAX_ROWS = 200
GRID_SIZE = 15
WHOLE_NUMBER_FIELDS = {
    "grid_x", "grid_y", "hazard_severity_1_5", "obstruction_level_0_3", "hour_of_day",
}
# These had gaps in the training data, so the model's imputer can fill them.
OPTIONAL_FIELDS = {"crowd_density_pct", "equipment_health_pct", "obstruction_level_0_3"}
MISSING_WORDS = {"", "na", "n/a", "nan", "null", "none", "unknown"}
INCIDENT_TYPES = [
    "Structural", "Medical", "Crowd_Control", "Electrical", "Equipment_Failure", "Fire_Smoke",
]
FIELDS = ["incident_id", "incident_type", *module2.NUMERIC_COLUMNS]


def _text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def parse_csv(text: str) -> dict:
    """Read CSV text into rows of strings. Raises ValueError for an unusable file."""
    text = text.lstrip("﻿")
    if not text.strip():
        raise ValueError("The CSV file is empty.")

    header = text.strip().splitlines()[0]
    delimiter = max([",", ";", "\t"], key=header.count)
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    try:
        lines = [line for line in reader if any(cell.strip() for cell in line)]
    except csv.Error as error:
        raise ValueError(f"The CSV file could not be read: {error}") from error

    columns = [name.strip().lower() for name in lines[0]]
    repeated = sorted({name for name in columns if name and columns.count(name) > 1})
    if repeated:
        raise ValueError("Repeated column(s): " + ", ".join(repeated))

    required = [name for name in FIELDS if name not in OPTIONAL_FIELDS and name != "incident_id"]
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError("Missing column(s): " + ", ".join(missing))
    if len(lines) == 1:
        raise ValueError("The CSV file has a header but no incidents.")
    if len(lines) - 1 > MAX_ROWS:
        raise ValueError(f"The CSV file has {len(lines) - 1} incidents; the limit is {MAX_ROWS}.")

    rows = []
    for line_number, line in enumerate(lines[1:], start=1):
        if len(line) > len(columns):
            raise ValueError(f"Row {line_number} has {len(line)} values but the header has {len(columns)}.")
        record = dict(zip(columns, line))
        rows.append({name: _text(record.get(name)) for name in FIELDS})

    return {"rows": rows, "ignored_columns": [c for c in columns if c and c not in FIELDS]}


def validate_rows(rows: list) -> tuple:
    """
    Check every row and collect ALL problems (not just the first).

    Returns (valid rows, errors, warnings). Each valid row keeps its
    original position in "row" (starting at 1).
    """
    types = {name.lower(): name for name in INCIDENT_TYPES}
    valid, errors, warnings, used_ids = [], [], [], set()

    for number, row in enumerate(rows, start=1):
        problems, blanks, clean = [], [], {"row": number}

        incident_id = _text(row.get("incident_id")) or f"USER-{number:03d}"
        if len(incident_id) > 40:
            problems.append(("incident_id", "must be 40 characters or fewer"))
        elif incident_id.lower() in used_ids:
            problems.append(("incident_id", f"'{incident_id}' is used more than once"))
        used_ids.add(incident_id.lower())
        clean["incident_id"] = incident_id

        type_text = _text(row.get("incident_type"))
        type_key = type_text.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
        if type_key in types:
            clean["incident_type"] = types[type_key]
        else:
            shown = f"'{type_text}' is not valid" if type_text else "is missing"
            problems.append(("incident_type", f"{shown}; use one of: {', '.join(INCIDENT_TYPES)}"))

        for field in module2.NUMERIC_COLUMNS:
            text = _text(row.get(field))
            low, high = module2.VALID_RANGES[field]
            if text.lower() in MISSING_WORDS:
                if field in OPTIONAL_FIELDS:
                    clean[field] = np.nan
                    blanks.append(field)
                else:
                    problems.append((field, "is missing"))
                continue
            try:
                value = float(text)
            except ValueError:
                problems.append((field, f"'{text}' is not a number"))
                continue
            if not (low <= value <= high):
                problems.append((field, f"must be {low}-{high} (got {text})"))
            elif field in WHOLE_NUMBER_FIELDS and value != int(value):
                problems.append((field, f"must be a whole number (got {text})"))
            else:
                clean[field] = value

        if problems:
            errors.extend({"row": number, "field": field, "message": f"Row {number}: {field} {message}"}
                          for field, message in problems)
        else:
            valid.append(clean)
            if blanks:
                warnings.append({"row": number, "message": f"Row {number}: {', '.join(blanks)} left blank, "
                                 "so the typical training value was used"})
    return valid, errors, warnings


def validate_grid(grid, start) -> tuple:
    """Check the station map and the vehicle's start cell."""
    if len(grid) != GRID_SIZE or any(len(row) != GRID_SIZE for row in grid):
        raise ValueError(f"The station map must be {GRID_SIZE}x{GRID_SIZE}.")
    array = np.array(grid, dtype=np.int64)
    if not np.all((array == -1) | ((array >= 1) & (array <= 50))):
        raise ValueError("Map cells must be -1 (wall) or a cost from 1 to 50.")
    if not all(0 <= value < GRID_SIZE for value in start):
        raise ValueError("The start cell is outside the station map.")
    start = (int(start[0]), int(start[1]))
    if array[start] == -1:
        raise ValueError("The start cell cannot be a wall.")
    return array, start


def analyse(rows: list, grid, start) -> dict:
    """Validate the input, run the trained models and plan the routes."""
    import module3_neural_network as module3

    if not rows:
        raise ValueError("Add at least one incident first.")
    if len(rows) > MAX_ROWS:
        raise ValueError(f"The limit is {MAX_ROWS} incidents at a time.")
    grid, start = validate_grid(grid, start)
    valid, errors, warnings = validate_rows(rows)

    output = {"errors": errors, "warnings": warnings, "incidents": [], "plan": None,
              "grid": grid.tolist(), "start": start, "submitted": len(rows)}
    if not valid:
        return output

    needed = ["best_risk_model.keras", "module3_preprocessor.joblib", "module3_knn_regressor.joblib",
              "location_kmeans.joblib", "location_scaler.joblib"]
    missing = [name for name in needed if not (module2.MODEL_DIR / name).exists()]
    if missing:
        raise FileNotFoundError("Saved models are missing: " + ", ".join(missing)
                                + ". Run Module 2 and train Module 3 first.")

    data = pd.DataFrame(valid)
    probabilities = module3.predict_risk_probabilities(data)
    data["predicted_risk"] = np.argmax(probabilities, axis=1)
    regressor = joblib.load(module2.MODEL_DIR / "module3_knn_regressor.joblib")
    data["predicted_response_time"] = regressor.predict(data[module3.FEATURE_COLUMNS])
    kmeans = joblib.load(module2.MODEL_DIR / "location_kmeans.joblib")
    scaler = joblib.load(module2.MODEL_DIR / "location_scaler.joblib")
    data["cluster"] = kmeans.predict(scaler.transform(data[["grid_x", "grid_y"]]))

    plan = module4.plan_user_incidents(grid, start, data)
    incidents = data.to_dict(orient="records")
    for index, incident in enumerate(incidents):
        incident["risk_level"] = module3.RISK_CLASSES[int(incident["predicted_risk"])]
        incident["probabilities"] = dict(zip(module3.RISK_CLASSES, probabilities[index].tolist()))
        incident.update(plan["routes"][incident["incident_id"]])
    incidents.sort(key=lambda item: item["priority"] if item["priority"] else math.inf)

    output["incidents"] = incidents
    output["plan"] = {key: plan[key] for key in ("predicted_map", "tour", "tour_cost")}
    return output
