# Student name: [Your Name]
# Roll number: [Your Roll Number]
# Course code: [Your Course Code]
# Module description: Integrated railway risk-aware routing and replanning

"""
Module 4 - Integrated smart system.

It connects the other modules:
    - Module 1: station layouts and the A* search algorithm
    - Module 2: the cleaned railway incident dataset
    - Module 3: the saved best ANN (risk level) and KNN regressor
      (response time)

How the integration works:
    1. Real incidents from the dataset are placed on the station grid at
       their (grid_x, grid_y) location.
    2. The saved ANN predicts each incident's risk level from its features.
       The agent never sees the true risk label while planning.
    3. "Without predictions": A* plans using travel cost only.
       "With predictions":    A* plans using travel cost + a penalty for
                              cells the ANN predicts are Medium/High risk.
    4. Both routes are then scored against the TRUE risk labels:
       hazardous cells slow the vehicle down and expose it to risk.
    5. During each run a new obstruction appears on the route and the
       agent re-plans from its current position.
"""

import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import module1_agent_search as module1
import module2_data_ml as module2


# All paths are relative to this file, so the program works from any folder.
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"

RISK_CLASSES = ["Low", "Medium", "High"]

# Extra planning cost the agent adds to a cell it PREDICTS is risky.
PREDICTED_RISK_PENALTY = {0: 0, 1: 4, 2: 10}
# Extra minutes the vehicle actually loses in a cell that IS risky
# (crowds, smoke, damaged equipment slow the response down).
ACTUAL_RISK_DELAY = {0: 0, 1: 2, 2: 6}

# Performance measure (same structure as Module 1):
# 100 per target reached - response time - 5 per re-plan - 10 per risk point
SUCCESS_REWARD = 100
REPLAN_PENALTY = 5
RISK_PENALTY = 10

INCIDENTS_PER_RUN = 60
RUNS_PER_LAYOUT = 10


# --------------------------------------------------------------------------
# Predictions from the saved Module 3 models
# --------------------------------------------------------------------------

def load_incidents_with_predictions() -> pd.DataFrame:
    """
    Load the cleaned dataset and add the ANN's predicted risk and the KNN
    regressor's predicted response time to every incident.
    """
    import module3_neural_network as module3

    required = [
        MODEL_DIR / "best_risk_model.keras",
        MODEL_DIR / "module3_preprocessor.joblib",
        MODEL_DIR / "module3_knn_regressor.joblib",
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Saved Module 3 models are missing: "
            + ", ".join(missing)
            + ". Run Module 3 first."
        )

    incidents = module2.load_prepared_data()
    probabilities = module3.predict_risk_probabilities(incidents)
    incidents["predicted_risk"] = np.argmax(probabilities, axis=1)
    incidents["actual_risk"] = incidents["risk_level"].map(
        {name: number for number, name in enumerate(RISK_CLASSES)}
    )

    regressor = joblib.load(MODEL_DIR / "module3_knn_regressor.joblib")
    incidents["predicted_response_time"] = regressor.predict(
        incidents[module3.FEATURE_COLUMNS]
    )

    accuracy = float((incidents["predicted_risk"] == incidents["actual_risk"]).mean())
    print(f"ANN risk predictions made for {len(incidents)} incidents "
          f"(agreement with true labels: {accuracy:.1%})")
    print("Note: this includes incidents the ANN was trained on. Its unbiased "
          "accuracy is the Module 3 test-set result.")
    return incidents


# --------------------------------------------------------------------------
# Station map with hazards
# --------------------------------------------------------------------------

def get_layouts() -> dict:
    """Use the same three layouts as Module 1."""
    return module1.create_layouts()


def place_incidents(
    grid: np.ndarray,
    incidents: pd.DataFrame,
    start: tuple,
    goal: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build two risk maps (0 = Low, 1 = Medium, 2 = High) from incidents.

    actual_map    - from the true labels (used only to score routes)
    predicted_map - from the ANN predictions (used by the agent to plan)

    If several incidents share a cell, the highest risk is kept.
    """
    actual_map = np.zeros(grid.shape, dtype=int)
    predicted_map = np.zeros(grid.shape, dtype=int)

    for _, incident in incidents.iterrows():
        cell = (int(incident["grid_y"]), int(incident["grid_x"]))
        if grid[cell] == -1 or cell in (start, goal):
            continue
        actual_map[cell] = max(actual_map[cell], int(incident["actual_risk"]))
        predicted_map[cell] = max(predicted_map[cell], int(incident["predicted_risk"]))

    return actual_map, predicted_map


def make_planning_grid(grid: np.ndarray, predicted_map=None) -> np.ndarray:
    """
    Grid used by A*.

    Without predictions: the plain travel-cost grid.
    With predictions:    travel cost + penalty for predicted-risk cells.
    Blocked cells (-1) stay blocked.
    """
    planning = grid.copy()
    if predicted_map is not None:
        free = planning != -1
        penalty = np.vectorize(PREDICTED_RISK_PENALTY.get)(predicted_map)
        planning[free] = planning[free] + penalty[free]
    return planning


def evaluate_route(grid: np.ndarray, actual_map: np.ndarray, path) -> dict:
    """
    Score a route against the TRUE risk.

    response time = travel cost + delay in cells that are actually risky
    risk encountered = sum of true risk levels on the route (Medium 1, High 2)
    """
    if path is None:
        return {"response_time": np.nan, "risk": 0, "travel_cost": np.nan}

    travel_cost = 0.0
    delay = 0.0
    risk = 0
    for cell in path[1:]:
        travel_cost += float(grid[cell])
        delay += ACTUAL_RISK_DELAY[int(actual_map[cell])]
        risk += int(actual_map[cell])

    return {
        "response_time": travel_cost + delay,
        "risk": risk,
        "travel_cost": travel_cost,
    }


def performance_measure(success: int, response_time: float, replans: int, risk: int) -> float:
    """100 * success - response time - 5 * re-plans - 10 * risk encountered."""
    response_time = 0.0 if np.isnan(response_time) else response_time
    return (
        SUCCESS_REWARD * success
        - response_time
        - REPLAN_PENALTY * replans
        - RISK_PENALTY * risk
    )


# --------------------------------------------------------------------------
# Route execution with dynamic re-planning
# --------------------------------------------------------------------------

def drive_with_replanning(
    planning_grid: np.ndarray,
    start: tuple,
    goal: tuple,
    rng: random.Random,
) -> dict:
    """
    Follow the A* route. Part-way along, a new obstruction appears on the
    cell just ahead. The agent marks it blocked and re-plans from where it
    currently is (not from the start).
    """
    planning_grid = planning_grid.copy()
    first_plan = module1.astar_search(planning_grid, start, goal)["path"]
    if first_plan is None:
        return {"path": None, "replans": 0, "new_block": None, "first_plan": None}

    replans = 0
    new_block = None
    driven = [start]

    # Only add an obstruction if the route is long enough to have a cell
    # strictly between the start and the goal.
    if len(first_plan) > 3:
        stop_index = rng.randint(1, len(first_plan) - 3)
        driven = first_plan[: stop_index + 1]
        new_block = first_plan[stop_index + 1]
        planning_grid[new_block] = -1
        replans = 1

        rest = module1.astar_search(planning_grid, driven[-1], goal)["path"]
        if rest is None:
            return {"path": None, "replans": replans,
                    "new_block": new_block, "first_plan": first_plan,
                    "driven": driven}
        path = driven + rest[1:]
    else:
        path = first_plan

    return {"path": path, "replans": replans, "new_block": new_block,
            "first_plan": first_plan, "driven": driven}


# --------------------------------------------------------------------------
# Demonstrations
# --------------------------------------------------------------------------

def plot_routes(grid, actual_map, routes: dict, start, goal, title, filename, extra=None):
    """Draw the grid, true hazards, blocked cells and one or more routes."""
    plt.figure(figsize=(8, 7))
    display = np.where(grid == -1, np.nan, grid).astype(float)
    plt.imshow(display, cmap="Greys", origin="upper", alpha=0.35)

    for level, colour, label in ((1, "orange", "Actual Medium risk"),
                                 (2, "red", "Actual High risk")):
        cells = np.argwhere(actual_map == level)
        if len(cells):
            plt.scatter(cells[:, 1], cells[:, 0], c=colour, s=120,
                        marker="s", alpha=0.6, label=label)

    blocked = np.argwhere(grid == -1)
    if len(blocked):
        plt.scatter(blocked[:, 1], blocked[:, 0], c="black", s=90,
                    marker="X", label="Blocked")

    styles = ["b-o", "g-o", "m--"]
    for (label, path), style in zip(routes.items(), styles):
        if path:
            plt.plot([p[1] for p in path], [p[0] for p in path], style,
                     linewidth=2, markersize=3, label=label)

    if extra is not None:
        plt.scatter(extra[1], extra[0], c="purple", s=220, marker="X",
                    label="New obstruction")

    plt.scatter(start[1], start[0], c="blue", s=160, marker="s", label="Start")
    plt.scatter(goal[1], goal[0], c="gold", s=220, marker="*", label="Goal")
    plt.title(title)
    plt.xlabel("Grid X (column)")
    plt.ylabel("Grid Y (row)")
    plt.legend(loc="upper right", fontsize=7)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.show()
    plt.close()


def compare_routing(incidents: pd.DataFrame = None) -> dict:
    """Compare a plain shortest route with a prediction-based risk-aware route."""
    if incidents is None:
        incidents = load_incidents_with_predictions()

    layout = get_layouts()["Normal Layout"]
    grid, start, goal = layout["grid"], layout["start"], layout["goal"]
    sample = incidents.sample(n=INCIDENTS_PER_RUN, random_state=0)
    actual_map, predicted_map = place_incidents(grid, sample, start, goal)

    plain = module1.astar_search(make_planning_grid(grid), start, goal)
    aware = module1.astar_search(make_planning_grid(grid, predicted_map), start, goal)

    rows = []
    for name, result in (("Normal shortest route", plain), ("Risk-aware route (ANN)", aware)):
        score = evaluate_route(grid, actual_map, result["path"])
        rows.append({
            "Route": name,
            "Length": result["path_length"],
            "Travel cost": score["travel_cost"],
            "Actual response time": score["response_time"],
            "Risk encountered": score["risk"],
        })
    table = pd.DataFrame(rows)

    print("\nNormal versus Risk-Aware Routing (Normal Layout, 60 incidents)")
    print(table.to_string(index=False))

    plot_routes(
        grid, actual_map,
        {"Normal route": plain["path"], "Risk-aware route": aware["path"]},
        start, goal,
        "Normal and Risk-Aware Railway Routes",
        "normal_and_risk_aware_routes.png",
    )

    return {"table": table, "grid": grid, "start": start, "goal": goal,
            "plain": plain, "risk_aware": aware,
            "actual_map": actual_map, "predicted_map": predicted_map}


def target_score(target: dict) -> float:
    """Priority score: higher risk first, then quicker and closer incidents."""
    return round(
        20 * target["predicted_risk"]
        - 0.5 * target["response_time"]
        - 0.5 * target["route_cost"],
        2,
    )


def select_next_target(targets: list) -> dict:
    """
    Rank incidents and choose which one to respond to first.

    score = 20 * predicted risk (0-2) - 0.5 * predicted response time
            - 0.5 * route cost from the vehicle
    Higher predicted risk raises the priority; a longer predicted response
    time or a longer route lowers it.
    """
    for target in targets:
        target["score"] = target_score(target)

    targets_sorted = sorted(targets, key=lambda target: target["score"], reverse=True)

    table = pd.DataFrame(targets_sorted)
    table["predicted_risk"] = table["predicted_risk"].map(dict(enumerate(RISK_CLASSES)))
    print("\nMulti-target planning (ranked):")
    print(table.to_string(index=False))
    print("Selected next target:", targets_sorted[0]["name"])
    return targets_sorted[0]


def demonstrate_multi_target(incidents: pd.DataFrame = None) -> list:
    """Pick four real incidents, rank them, and plan a route visiting all of them."""
    if incidents is None:
        incidents = load_incidents_with_predictions()

    layout = get_layouts()["Normal Layout"]
    grid, start = layout["grid"], layout["start"]

    # One incident of each risk level the ANN predicts, plus one more.
    candidates = []
    for level in (2, 1, 0, 0):
        pool = incidents[(incidents["predicted_risk"] == level)
                         & ~incidents["incident_id"].isin([c["name"] for c in candidates])]
        pool = pool[[grid[int(r.grid_y), int(r.grid_x)] != -1 for r in pool.itertuples()]]
        if pool.empty:
            continue
        row = pool.sample(n=1, random_state=len(candidates)).iloc[0]
        cell = (int(row["grid_y"]), int(row["grid_x"]))
        route = module1.astar_search(grid, start, cell)
        candidates.append({
            "name": row["incident_id"],
            "type": row["incident_type"],
            "cell": cell,
            "predicted_risk": int(row["predicted_risk"]),
            "response_time": round(float(row["predicted_response_time"]), 2),
            "route_cost": route["cost"] if route["path"] else 999,
        })

    if not candidates:
        print("No reachable incidents were found.")
        return []

    select_next_target(candidates)
    order = sorted(candidates, key=lambda target: target["score"], reverse=True)

    position = start
    total_cost = 0.0
    print("\nVisiting order:")
    for target in order:
        leg = module1.astar_search(grid, position, target["cell"])
        if leg["path"] is None:
            print(f"  {target['name']} at {target['cell']}: unreachable, skipped")
            continue
        total_cost += leg["cost"]
        print(f"  {target['name']} at {target['cell']}: leg cost {leg['cost']:.0f}")
        position = target["cell"]
    print("Total multi-target route cost:", total_cost)
    return order


def plan_user_incidents(grid: np.ndarray, start: tuple, incidents: pd.DataFrame) -> dict:
    """
    Rank user-supplied incidents and plan routes to each of them.

    All risk values here are PREDICTED, because user data has no true labels.
    An incident's own cell is not counted as a hazard on the way to it: the
    vehicle has to enter that cell on every possible route.
    """
    predicted_map = np.zeros(grid.shape, dtype=int)
    for incident in incidents.itertuples():
        cell = (int(incident.grid_y), int(incident.grid_x))
        if grid[cell] != -1 and cell != start:
            predicted_map[cell] = max(predicted_map[cell], int(incident.predicted_risk))

    def hazards_towards(cell):
        hazards = predicted_map.copy()
        hazards[cell] = 0
        return hazards

    def summarise(path, hazards):
        score = evaluate_route(grid, hazards, path)
        return {"path": path, "travel_cost": score["travel_cost"],
                "predicted_time": score["response_time"], "predicted_risk": score["risk"]}

    routes, ranking = {}, []
    for incident in incidents.itertuples():
        cell = (int(incident.grid_y), int(incident.grid_x))
        hazards = hazards_towards(cell)
        normal_path = aware_path = None
        if grid[cell] != -1:
            normal_path = module1.astar_search(make_planning_grid(grid), start, cell)["path"]
            aware_path = module1.astar_search(make_planning_grid(grid, hazards), start, cell)["path"]

        reachable = normal_path is not None
        route = {
            "cell": cell, "reachable": reachable, "priority": None, "score": None,
            "reason": None if reachable else (
                "The incident is inside a wall" if grid[cell] == -1 else "No route exists from the start"),
            "normal": summarise(normal_path, hazards) if reachable else None,
            "risk_aware": summarise(aware_path, hazards) if reachable else None,
        }
        if reachable:
            route["score"] = target_score({
                "predicted_risk": int(incident.predicted_risk),
                "response_time": float(incident.predicted_response_time),
                "route_cost": route["normal"]["travel_cost"],
            })
            ranking.append(incident.incident_id)
        routes[incident.incident_id] = route

    ranking.sort(key=lambda name: routes[name]["score"], reverse=True)

    # Visit every reachable incident in priority order, avoiding predicted risk.
    tour, tour_cost, position = [start], 0.0, start
    for number, name in enumerate(ranking, start=1):
        routes[name]["priority"] = number
        cell = routes[name]["cell"]
        leg = module1.astar_search(
            make_planning_grid(grid, hazards_towards(cell)), position, cell)["path"]
        tour.extend(leg[1:])
        tour_cost += module1.calculate_path_cost(leg, grid)
        position = cell

    return {"routes": routes, "predicted_map": predicted_map.tolist(),
            "tour": tour, "tour_cost": tour_cost}


def demonstrate_replanning(current_index: int = 6) -> dict:
    """
    Show automatic re-planning from the current position after a new block.

    current_index: how many steps along the route the vehicle has travelled
    when the obstruction ahead is detected.
    """
    layout = get_layouts()["Normal Layout"]
    grid, start, goal = layout["grid"], layout["start"], layout["goal"]

    route_before = module1.astar_search(grid, start, goal)["path"]
    print("\nRoute before re-planning:")
    print(route_before)

    if route_before is None or len(route_before) < 4:
        print("Route too short to demonstrate re-planning.")
        return {}

    current_index = max(1, min(int(current_index), len(route_before) - 3))
    current = route_before[current_index]
    new_blocked_cell = route_before[current_index + 1]
    print("Vehicle has reached:", current)
    print("Newly detected blocked cell ahead:", new_blocked_cell)

    updated = grid.copy()
    updated[new_blocked_cell] = -1

    rest = module1.astar_search(updated, current, goal)["path"]
    print("Agent action: update internal map and re-plan from the current position")
    before_cost = module1.calculate_path_cost(route_before, grid)
    after_cost = None
    if rest is None:
        print("No path exists - alert control room")
        route_after = None
    else:
        route_after = route_before[:current_index] + rest
        print("Route after re-planning:")
        print(route_after)
        after_cost = module1.calculate_path_cost(route_after, updated)
        print(f"Planned cost before: {before_cost:.0f}, actual cost after re-planning: {after_cost:.0f}")

    plot_routes(
        updated, np.zeros(grid.shape, dtype=int),
        {"Route before obstacle": route_before, "Route after re-planning": route_after},
        start, goal,
        "Automatic Re-planning",
        "automatic_replanning.png",
        extra=new_blocked_cell,
    )

    return {
        "grid": updated,
        "start": start,
        "goal": goal,
        "route_before": route_before,
        "route_after": route_after,
        "current": current,
        "current_index": current_index,
        "new_block": new_blocked_cell,
        "cost_before": before_cost,
        "cost_after": after_cost,
    }


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------

def run_experiments(incidents: pd.DataFrame = None) -> pd.DataFrame:
    """Run 10 experiments on each of the 3 layouts, with and without predictions."""
    if incidents is None:
        incidents = load_incidents_with_predictions()

    experiment_results = []
    for layout_number, (layout_name, layout) in enumerate(get_layouts().items(), start=1):
        grid, start, goal = layout["grid"], layout["start"], layout["goal"]

        for run_number in range(1, RUNS_PER_LAYOUT + 1):
            seed = 1000 * layout_number + run_number
            sample = incidents.sample(n=INCIDENTS_PER_RUN, random_state=seed)
            actual_map, predicted_map = place_incidents(grid, sample, start, goal)

            modes = {
                "Without predictions": make_planning_grid(grid),
                "With predictions": make_planning_grid(grid, predicted_map),
            }
            for mode, planning_grid in modes.items():
                # Same random event position rule for both modes in a run.
                drive = drive_with_replanning(planning_grid, start, goal, random.Random(seed))
                success = int(drive["path"] is not None)

                score_grid = grid.copy()
                if drive["new_block"] is not None:
                    score_grid[drive["new_block"]] = -1
                score = evaluate_route(score_grid, actual_map, drive["path"])

                experiment_results.append({
                    "Layout": layout_number,
                    "Layout name": layout_name,
                    "Run": run_number,
                    "Mode": mode,
                    "Successful target": success,
                    "Path length": len(drive["path"]) if success else 0,
                    "Travel cost": score["travel_cost"],
                    "Response time": score["response_time"],
                    "Risk encountered": score["risk"],
                    "Re-plans": drive["replans"],
                    "Performance": round(performance_measure(
                        success, score["response_time"], drive["replans"], score["risk"]
                    ), 2),
                })

    results = pd.DataFrame(experiment_results)
    print("\nModule 4 Experiment Results")
    print(results.drop(columns=["Layout name"]).to_string(index=False))

    summary = results.groupby(["Layout", "Layout name", "Mode"], sort=False).agg({
        "Successful target": "mean",
        "Path length": "mean",
        "Travel cost": "mean",
        "Response time": "mean",
        "Risk encountered": "mean",
        "Re-plans": "mean",
        "Performance": "mean",
    }).reset_index()

    print("\nAverage Results (response time is n/a when the goal is unreachable)")
    print(summary.round(2).to_string(index=False, na_rep="n/a"))

    plt.figure(figsize=(10, 6))
    layout_labels = summary.drop_duplicates("Layout")["Layout name"].tolist()
    bar_width = 0.38
    positions = np.arange(len(layout_labels))
    for offset, mode in ((-bar_width / 2, "Without predictions"),
                         (bar_width / 2, "With predictions")):
        values = summary[summary["Mode"] == mode]["Performance"].to_numpy()
        plt.bar(positions + offset, values, width=bar_width, label=mode)
    plt.xticks(positions, layout_labels)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Average performance score")
    plt.title("Agent Performance: With and Without ANN Risk Predictions")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "prediction_experiment_results.png")
    plt.show()
    plt.close()

    results.to_csv(OUTPUT_DIR / "module4_experiment_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "module4_experiment_summary.csv", index=False)
    return results


def run_module4() -> dict:
    """Run all Module 4 functions."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    incidents = load_incidents_with_predictions()
    routing = compare_routing(incidents)
    targets = demonstrate_multi_target(incidents)
    demonstrate_replanning()
    experiment_results = run_experiments(incidents)

    print("\nModule 4 completed.")
    print("Graphs and CSV results saved in:", OUTPUT_DIR)

    return {
        "routing": routing,
        "selected_target": targets[0] if targets else None,
        "experiments": experiment_results,
    }


def module4_menu() -> dict:
    """Simple console menu for Module 4."""
    print("\nMODULE 4 - Integrated Smart System")
    print("1. Run complete Module 4")
    print("2. Compare normal and risk-aware routing")
    print("3. Demonstrate multi-target planning")
    print("4. Demonstrate automatic re-planning")
    print("5. Run 10 experiments on 3 layouts")
    print("6. Exit")

    try:
        choice = input("Enter choice: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("Exiting Module 4.")
        return {}

    OUTPUT_DIR.mkdir(exist_ok=True)
    if choice == "1":
        return run_module4()
    if choice == "2":
        return {"routing": compare_routing()}
    if choice == "3":
        return {"targets": demonstrate_multi_target()}
    if choice == "4":
        demonstrate_replanning()
        return {}
    if choice == "5":
        return {"experiments": run_experiments()}
    if choice == "6":
        print("Exiting Module 4.")
        return {}

    print("Invalid choice.")
    return {}


if __name__ == "__main__":
    module4_menu()
