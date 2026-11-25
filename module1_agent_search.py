# Student name: [Your Name]
# Roll number: [Your Roll Number]
# Course code: [Your Course Code]
# Module description: Railway station model-based agent and grid-search algorithms

"""
Module 1:
Model-based reflex agent, weighted grid, BFS, DFS, A*,
search comparison, visualisation, and five agent demonstrations.
"""

from __future__ import annotations

import heapq
import time
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Position = Tuple[int, int]
Grid = np.ndarray

# All paths are relative to this file, so the program works from any folder.
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
DATASET_FILENAME = "railway_station_incident_response_dataset.csv"

# Performance measure used by the agent (same weights as Module 4).
SUCCESS_REWARD = 100
REPLAN_PENALTY = 5
REQUIRED_COLUMNS = {
    "incident_id",
    "grid_x",
    "grid_y",
    "incident_type",
    "crowd_density_pct",
    "hazard_severity_1_5",
    "equipment_health_pct",
    "obstruction_level_0_3",
    "hour_of_day",
    "risk_level",
    "response_time_min",
}


def find_dataset_path() -> Path:
    """Find the CSV in the project folder."""
    project_path = PROJECT_DIR / DATASET_FILENAME
    if project_path.exists():
        return project_path

    raise FileNotFoundError(
        f"Could not find {DATASET_FILENAME}. Copy it beside module1_agent_search.py "
        "and run the program again."
    )


def load_railway_dataset(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load and validate the supplied railway incident CSV."""
    path = csv_path if csv_path is not None else find_dataset_path()

    try:
        dataset = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise ValueError(f"Unable to load railway dataset: {error}") from error

    missing_columns = REQUIRED_COLUMNS.difference(dataset.columns)
    if missing_columns:
        raise ValueError(
            "The dataset does not match the required railway schema. "
            f"Missing columns: {sorted(missing_columns)}"
        )

    # Grid coordinates use grid_x as the column and grid_y as the row.
    for coordinate in ("grid_x", "grid_y"):
        dataset[coordinate] = pd.to_numeric(dataset[coordinate], errors="coerce")

    invalid_coordinates = dataset[
        dataset["grid_x"].isna()
        | dataset["grid_y"].isna()
        | dataset["grid_x"].lt(0)
        | dataset["grid_x"].gt(14)
        | dataset["grid_y"].lt(0)
        | dataset["grid_y"].gt(14)
    ]
    if not invalid_coordinates.empty:
        raise ValueError(
            "The dataset contains coordinates outside the 15x15 railway grid."
        )

    if dataset["risk_level"].dtype == "object":
        valid_risk = {"Low", "Medium", "High"}
        if not set(dataset["risk_level"].dropna()).issubset(valid_risk):
            raise ValueError("risk_level must contain only Low, Medium, or High.")

    print(f"Dataset loaded: {path}")
    print("Dataset shape:", dataset.shape)
    print("Required schema: all railway columns are present")
    print("Missing values:")
    print(dataset.isna().sum().to_string())
    print()
    return dataset


def csv_incident_to_position(row: pd.Series) -> Position:
    """Convert a CSV record to (grid_y, grid_x) for NumPy grid indexing."""
    grid_x = int(row["grid_x"])
    grid_y = int(row["grid_y"])
    return (grid_y, grid_x)


def create_layout_from_dataset(
    dataset: pd.DataFrame,
    grid_size: int = 15,
) -> Tuple[Grid, Position, Position, List[pd.Series]]:
    """
    Create a 15x15 weighted grid using real CSV coordinates.

    The grid cost combines crowd density, hazard severity, obstruction,
    and equipment health. Risk level is used to mark a high-risk area.
    """
    if grid_size < 10:
        raise ValueError("The railway grid must be at least 10x10.")

    grid = create_weighted_grid(grid_size, grid_size)
    incident_rows: List[pd.Series] = []

    for _, row in dataset.iterrows():
        if pd.isna(row["grid_x"]) or pd.isna(row["grid_y"]):
            continue

        position = csv_incident_to_position(row)
        row_index, col_index = position

        if not (0 <= row_index < grid_size and 0 <= col_index < grid_size):
            continue

        incident_rows.append(row)

        crowd = row["crowd_density_pct"] if pd.notna(row["crowd_density_pct"]) else 50
        hazard = row["hazard_severity_1_5"] if pd.notna(row["hazard_severity_1_5"]) else 3
        obstruction = (
            row["obstruction_level_0_3"]
            if pd.notna(row["obstruction_level_0_3"])
            else 0
        )

        cost = 1
        if crowd >= 70:
            cost += 2
        if hazard >= 4:
            cost += 3
        if obstruction >= 2:
            cost += 2

        risk = row["risk_level"]
        if risk == "High":
            cost += 4
        elif risk == "Medium":
            cost += 2

        # Several incidents can share one cell: keep the most expensive one
        # instead of letting the last CSV row overwrite the others.
        grid[row_index, col_index] = max(int(grid[row_index, col_index]), cost)

    start = (14, 0)
    goal = (0, 14)

    if grid[start[0], start[1]] == -1:
        grid[start[0], start[1]] = 1
    if grid[goal[0], goal[1]] == -1:
        grid[goal[0], goal[1]] = 1

    return grid, start, goal, incident_rows


class RailwayResponseAgent:
    """
    Model-based reflex agent for a railway station emergency response vehicle.

    Internal model (state):
        - a map of the station; blocked cells are unknown at the start and are
          added when the agent senses them (free-space assumption)
        - the current planned route to the target
        - visited cells, route cost so far and number of re-plans

    The agent only senses its four neighbouring cells. Its condition-action
    rules use that percept together with the internal model.
    """

    def __init__(
        self,
        start: Position,
        target: Position,
        station_map: Optional[Grid] = None,
    ) -> None:
        self.current_position = start
        self.target_position = target
        self.visited_cells: set[Position] = {start}
        self.discovered_blocked_cells: set[Position] = set()
        self.last_action = "started"
        self.path_taken: List[Position] = [start]
        self.planned_route: Optional[List[Position]] = None
        self.route_cost = 0.0
        self.blocked_replans = 0
        self.wait_count = 0
        self.reached_target = False
        self.gave_up = False
        # One entry per decision, used by the web UI to replay the episode.
        self.trace: List[Dict[str, object]] = []

        # The agent knows the movement cost of each cell, but not which cells
        # are blocked. Unknown blocks are treated as normal cost-1 cells.
        if station_map is None:
            station_map = create_weighted_grid(15, 15)
        self.internal_map = np.where(station_map == -1, 1, station_map).astype(np.int64)

    def update_internal_model(self, percept: Dict[Position, str]) -> bool:
        """Store newly sensed blocked cells. Return True if the map changed."""
        changed = False
        for position, value in percept.items():
            if value == "blocked" and position not in self.discovered_blocked_cells:
                self.discovered_blocked_cells.add(position)
                self.internal_map[position] = -1
                changed = True
        return changed

    def plan_route(self) -> Optional[List[Position]]:
        """Plan a route with A* on the agent's internal map."""
        result = astar_search(self.internal_map, self.current_position, self.target_position)
        return result["path"]

    def route_is_blocked(self) -> bool:
        """Check whether the stored route passes a cell now known to be blocked."""
        if not self.planned_route:
            return False
        return any(cell in self.discovered_blocked_cells for cell in self.planned_route)

    def sense_neighbours(self, grid: Grid) -> Dict[Position, str]:
        """
        Sense only the four immediately neighbouring cells.

        Possible percept values:
            - "blocked": the neighbouring cell is blocked
            - "goal": the neighbouring cell is the destination
            - "normal": a free neighbour
            - "out_of_bounds": outside the grid
        """
        row, col = self.current_position
        rows, cols = grid.shape
        neighbours: Dict[Position, str] = {}

        for next_position in get_neighbours(self.current_position):
            next_row, next_col = next_position
            if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                neighbours[next_position] = "out_of_bounds"
            elif grid[next_row, next_col] == -1:
                neighbours[next_position] = "blocked"
            elif next_position == self.target_position:
                neighbours[next_position] = "goal"
            else:
                neighbours[next_position] = "normal"

        return neighbours

    def choose_action(self, grid: Grid) -> str:
        """Choose and perform one action, and record it in the trace."""
        position_before = self.current_position
        percept = self.sense_neighbours(grid)
        replans_before = self.blocked_replans
        action = self._choose_action(grid)
        self.trace.append({
            "position_before": position_before,
            "position": self.current_position,
            "percept": {f"{r},{c}": value for (r, c), value in percept.items()},
            "action": action,
            "replanned": self.blocked_replans > replans_before,
            "discovered_blocked": sorted(self.discovered_blocked_cells),
            "planned_route": list(self.planned_route) if self.planned_route else [],
            "route_cost": self.route_cost,
        })
        return action

    def _choose_action(self, grid: Grid) -> str:
        """
        Apply condition-action rules using the neighbouring percept and internal state.

        Rules, checked in order:
            1. At the target                     -> alert: incident location reached
            2. Every neighbour blocked/outside   -> alert: no safe neighbouring move
            3. Target is a free neighbour        -> move to the target
            4. Route crosses a sensed block      -> re-plan on the internal map
            5. No route on the internal map      -> alert: target unreachable
            6. Otherwise                         -> follow the next step of the route
        """
        if self.current_position == self.target_position:
            self.reached_target = True
            self.last_action = "alert: incident location reached"
            return self.last_action

        percept = self.sense_neighbours(grid)
        self.update_internal_model(percept)

        if all(value in ("blocked", "out_of_bounds") for value in percept.values()):
            self.gave_up = True
            self.last_action = "alert: no safe neighbouring move"
            return self.last_action

        for position, value in percept.items():
            if value == "goal":
                self.planned_route = [self.current_position, position]
                self.move(position, grid)
                self.reached_target = True
                self.last_action = f"move to goal {position}"
                return self.last_action

        replanned = False
        if self.planned_route is None:
            self.planned_route = self.plan_route()
        elif self.route_is_blocked():
            self.blocked_replans += 1
            replanned = True
            self.planned_route = self.plan_route()

        if self.planned_route is None:
            self.gave_up = True
            self.last_action = "alert: target unreachable from the known map"
            return self.last_action

        next_position = self.planned_route[1]
        # Remove the cell we are leaving so the route always starts here.
        self.planned_route = self.planned_route[1:]
        self.move(next_position, grid)

        prefix = "re-plan and move to" if replanned else "move to"
        self.last_action = f"{prefix} {next_position}"
        return self.last_action

    def move(self, position: Position, grid: Optional[Grid] = None) -> None:
        """Update the agent state after a move."""
        self.current_position = position
        self.visited_cells.add(position)
        self.path_taken.append(position)
        if grid is not None:
            self.route_cost += float(grid[position])

    def run_episode(self, grid: Grid, max_steps: int = 300) -> Dict[str, object]:
        """Keep choosing actions until the target is reached or the agent stops."""
        steps = 0
        while steps < max_steps and not self.reached_target and not self.gave_up:
            self.choose_action(grid)
            steps += 1

        if self.current_position == self.target_position:
            self.reached_target = True
            self.last_action = "alert: incident location reached"

        return {
            "success": int(self.reached_target),
            "route_cost": self.route_cost,
            "steps": len(self.path_taken) - 1,
            "blocked_replans": self.blocked_replans,
            "final_action": self.last_action,
            "performance": self.calculate_performance(
                successful_targets=int(self.reached_target),
                route_cost=self.route_cost,
                blocked_replans=self.blocked_replans,
                wait_count=self.wait_count,
            ),
        }

    def print_status(self, grid: Grid) -> None:
        """Print the percept, the selected action, and the updated internal state."""
        percept = self.sense_neighbours(grid)
        action = self.choose_action(grid)

        print("Percept (before acting):", percept)
        print("Selected action:", action)
        print("Internal state (after acting):")
        print("  Current position:", self.current_position)
        print("  Visited cells:", sorted(self.visited_cells))
        print("  Discovered blocked cells:", sorted(self.discovered_blocked_cells))
        print("  Planned route:", self.planned_route)
        print("  Re-plans so far:", self.blocked_replans)
        print()

    def calculate_performance(
        self,
        successful_targets: int,
        route_cost: float,
        blocked_replans: int,
        wait_count: int,
    ) -> float:
        """
        Performance = 100 * successful_targets - route_cost
                      - 5 * blocked_replans - wait_count
        """
        return float(
            SUCCESS_REWARD * successful_targets
            - route_cost
            - (REPLAN_PENALTY * blocked_replans)
            - wait_count
        )


def get_neighbours(position: Position) -> List[Position]:
    """Return four-direction neighbouring positions."""
    row, col = position
    return [
        (row - 1, col),
        (row, col + 1),
        (row + 1, col),
        (row, col - 1),
    ]


def manhattan_distance(start: Position, goal: Position) -> int:
    """Calculate admissible Manhattan distance for four-direction movement."""
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])


def validate_positions(
    grid: Grid,
    start: Position,
    goal: Position,
    blocked: Iterable[Position],
) -> None:
    """Reject invalid start, goal, or blocked-cell positions."""
    rows, cols = grid.shape

    for name, position in (("start", start), ("goal", goal)):
        row, col = position
        if row < 0 or row >= rows or col < 0 or col >= cols:
            raise ValueError(f"{name} position {position} is outside the grid")

    if start in blocked:
        raise ValueError("The start position cannot be blocked")
    if goal in blocked:
        raise ValueError("The goal position cannot be blocked")


def create_weighted_grid(
    rows: int = 15,
    cols: int = 15,
    blocked_cells: Optional[Iterable[Position]] = None,
) -> Grid:
    """Create a weighted 15x15 grid. -1 means blocked."""
    grid = np.ones((rows, cols), dtype=np.int64)

    if blocked_cells:
        for row, col in blocked_cells:
            if 0 <= row < rows and 0 <= col < cols:
                grid[row, col] = -1

    # Higher values represent slower or more expensive railway areas.
    for row in range(rows):
        for col in range(cols):
            if grid[row, col] != -1:
                if (row + col) % 7 == 0:
                    grid[row, col] = 3
                elif (row * 2 + col) % 11 == 0:
                    grid[row, col] = 2
                else:
                    grid[row, col] = 1

    return grid


def reconstruct_path(
    came_from: Dict[Position, Optional[Position]],
    current: Position,
) -> List[Position]:
    """Reconstruct a path from the predecessor dictionary."""
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def calculate_path_cost(path: Sequence[Position], grid: Grid) -> float:
    """Return the sum of movement costs, excluding the starting cell."""
    return float(sum(grid[row, col] for row, col in path[1:] if grid[row, col] != -1))


def bfs_search(
    grid: Grid,
    start: Position,
    goal: Position,
) -> Dict[str, object]:
    """Breadth-first search implemented completely from scratch."""
    start_time = time.perf_counter()
    queue = deque([start])
    visited = {start}
    came_from: Dict[Position, Optional[Position]] = {start: None}
    nodes_expanded = 0
    explored: List[Position] = []

    if start == goal:
        elapsed = time.perf_counter() - start_time
        return {"path": [start], "path_length": 1,
                "cost": 0.0, "nodes_expanded": 1, "time": elapsed,
                "explored": [start]}

    while queue:
        current = queue.popleft()
        nodes_expanded += 1
        explored.append(current)

        if current == goal:
            break

        for neighbour in get_neighbours(current):
            row, col = neighbour
            if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
                if grid[row, col] != -1 and neighbour not in visited:
                    visited.add(neighbour)
                    came_from[neighbour] = current
                    queue.append(neighbour)
    else:
        return {"path": None, "path_length": 0,
                "cost": 0.0, "nodes_expanded": nodes_expanded,
                "time": time.perf_counter() - start_time,
                "explored": explored}

    path = reconstruct_path(came_from, goal)
    return {
        "path": path,
        "path_length": len(path),
        "cost": calculate_path_cost(path, grid),
        "nodes_expanded": nodes_expanded,
        "time": time.perf_counter() - start_time,
        "explored": explored,
    }


def dfs_search(
    grid: Grid,
    start: Position,
    goal: Position,
) -> Dict[str, object]:
    """Depth-first search implemented completely from scratch."""
    start_time = time.perf_counter()
    stack = [start]
    visited = {start}
    came_from: Dict[Position, Optional[Position]] = {start: None}
    nodes_expanded = 0
    reached_goal = False
    explored: List[Position] = []

    while stack:
        current = stack.pop()
        nodes_expanded += 1
        explored.append(current)

        if current == goal:
            reached_goal = True
            break

        # Reverse the list so upward/right/down/left are explored naturally.
        for neighbour in reversed(get_neighbours(current)):
            row, col = neighbour
            if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
                if grid[row, col] != -1 and neighbour not in visited:
                    visited.add(neighbour)
                    came_from[neighbour] = current
                    stack.append(neighbour)

    elapsed = time.perf_counter() - start_time
    if not reached_goal:
        return {"path": None, "path_length": 0,
                "cost": 0.0, "nodes_expanded": nodes_expanded, "time": elapsed,
                "explored": explored}

    path = reconstruct_path(came_from, goal)
    return {
        "path": path,
        "path_length": len(path),
        "cost": calculate_path_cost(path, grid),
        "nodes_expanded": nodes_expanded,
        "time": elapsed,
        "explored": explored,
    }


def astar_search(
    grid: Grid,
    start: Position,
    goal: Position,
) -> Dict[str, object]:
    """
    A* search implemented completely from scratch.

    It uses g(n) + h(n), where g(n) is the weighted path cost and h(n)
    is Manhattan distance times the minimum movement cost (1). With
    four-direction movement and no diagonal movement, Manhattan distance
    never overestimates the true minimum remaining cost, so the heuristic
    is admissible.
    """
    start_time = time.perf_counter()
    frontier: List[Tuple[int, int, Position]] = []
    counter = 0
    heapq.heappush(frontier, (manhattan_distance(start, goal), counter, start))

    came_from: Dict[Position, Optional[Position]] = {start: None}
    cost_so_far: Dict[Position, float] = {start: 0.0}
    closed: set[Position] = set()
    explored: List[Position] = []
    nodes_expanded = 0
    reached_goal = False

    while frontier:
        _, _, current = heapq.heappop(frontier)

        # A cell can be pushed more than once when a cheaper route is found.
        # Skip the older, more expensive copies so each cell is expanded once.
        if current in closed:
            continue
        closed.add(current)

        nodes_expanded += 1
        explored.append(current)
        if current == goal:
            reached_goal = True
            break

        for neighbour in get_neighbours(current):
            row, col = neighbour
            if row < 0 or row >= grid.shape[0] or col < 0 or col >= grid.shape[1]:
                continue

            movement_cost = int(grid[row, col])
            if movement_cost == -1:
                continue

            new_cost = cost_so_far[current] + movement_cost
            if neighbour not in cost_so_far or new_cost < cost_so_far[neighbour]:
                cost_so_far[neighbour] = new_cost
                came_from[neighbour] = current
                priority = new_cost + manhattan_distance(neighbour, goal)
                counter += 1
                heapq.heappush(frontier, (priority, counter, neighbour))

    elapsed = time.perf_counter() - start_time
    if not reached_goal:
        return {"path": None, "path_length": 0,
                "cost": 0.0, "nodes_expanded": nodes_expanded, "time": elapsed,
                "explored": explored}

    path = reconstruct_path(came_from, goal)
    return {
        "path": path,
        "path_length": len(path),
        "cost": cost_so_far[goal],
        "nodes_expanded": nodes_expanded,
        "time": elapsed,
        "explored": explored,
    }


def run_all_searches(
    grid: Grid,
    start: Position,
    goal: Position,
) -> Dict[str, Dict[str, object]]:
    """Run BFS, DFS, and A* on the same grid and compare results."""
    return {
        "BFS": bfs_search(grid, start, goal),
        "DFS": dfs_search(grid, start, goal),
        "A*": astar_search(grid, start, goal),
    }


def print_comparison_table(results: Dict[str, Dict[str, object]]) -> None:
    """Print the BFS, DFS, and A* comparison table."""
    print("\nSearch Algorithm Comparison")
    print("-" * 82)
    print(f"{'Algorithm':<12} {'Path Length':<14} {'Total Cost':<14} "
          f"{'Nodes Expanded':<16} {'Time (seconds)':<16}")
    print("-" * 82)

    for name, result in results.items():
        path_length = result["path_length"]
        cost = result["cost"]
        nodes = result["nodes_expanded"]
        execution_time = result["time"]
        print(f"{name:<12} {path_length:<14} {float(cost):<14.2f} "
              f"{nodes:<16} {float(execution_time):<16.6f}")

    valid_results = {
        name: result for name, result in results.items()
        if result["path"] is not None
    }

    if not valid_results:
        print("\nNo path exists")
        return

    best_name = min(
        valid_results,
        key=lambda name: (
            float(valid_results[name]["cost"]),
            int(valid_results[name]["path_length"]),
            float(valid_results[name]["nodes_expanded"]),
        ),
    )
    print(
        f"\nBest algorithm: {best_name}. It has the lowest total route cost "
        "for this weighted grid, with path length and nodes expanded considered."
    )


def visualise_grid(
    grid: Grid,
    start: Position,
    goal: Position,
    chosen_path: Optional[Sequence[Position]],
    title: str,
) -> None:
    """Draw the grid and highlight blocked cells and the selected path."""
    display = np.where(grid == -1, -1, grid)
    plt.figure(figsize=(8, 8))
    plt.imshow(display, cmap="coolwarm", origin="upper")

    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if grid[row, col] != -1:
                plt.text(col, row, str(grid[row, col]),
                         ha="center", va="center", fontsize=7)

    if chosen_path:
        path_rows = [position[0] for position in chosen_path]
        path_cols = [position[1] for position in chosen_path]
        plt.plot(path_cols, path_rows, color="green",
                 linewidth=3, marker="o", markersize=4,
                 label="A* selected path")

    plt.scatter(start[1], start[0], c="blue", s=180,
                marker="s", label="Start")
    plt.scatter(goal[1], goal[0], c="gold", s=180,
                marker="*", label="Goal")

    blocked_rows = [row for row in range(grid.shape[0])
                    for col in range(grid.shape[1])
                    if grid[row, col] == -1]
    blocked_cols = [col for row in range(grid.shape[0])
                    for col in range(grid.shape[1])
                    if grid[row, col] == -1]

    if blocked_rows:
        plt.scatter(blocked_cols, blocked_rows, c="black", s=90,
                    marker="X", label="Blocked")

    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.xticks(range(grid.shape[1]))
    plt.yticks(range(grid.shape[0]))
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right")
    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    safe_title = title.lower().replace(" ", "_").replace("*", "")
    plt.savefig(OUTPUT_DIR / f"{safe_title}.png", dpi=150)
    plt.show()
    plt.close()


def create_layouts() -> Dict[str, Dict[str, object]]:
    """Create normal, obstacle-heavy, and unreachable-goal layouts."""
    layouts: Dict[str, Dict[str, object]] = {}

    normal_blocked = {
        (3, 3), (3, 4), (3, 5), (7, 8),
        (7, 9), (8, 9), (10, 4),
    }
    layouts["Normal Layout"] = {
        "grid": create_weighted_grid(15, 15, normal_blocked),
        "start": (13, 1),
        "goal": (1, 13),
    }

    obstacle_blocked = set(normal_blocked)
    obstacle_blocked.update({
        (1, 6), (2, 6), (3, 6), (4, 6),
        (5, 6), (6, 6), (7, 6), (8, 6),
        (9, 6), (10, 6), (11, 6),
        (5, 9), (5, 10), (5, 11), (5, 12),
        (9, 1), (9, 2), (9, 3), (9, 4),
    })
    layouts["Obstacle-Heavy Layout"] = {
        "grid": create_weighted_grid(15, 15, obstacle_blocked),
        "start": (13, 1),
        "goal": (1, 13),
    }

    unreachable_blocked = {
        (0, 12), (0, 13), (0, 14),
        (1, 12), (1, 14),
        (2, 12), (2, 13), (2, 14),
    }
    unreachable_blocked.update(normal_blocked)
    layouts["Unreachable-Goal Layout"] = {
        "grid": create_weighted_grid(15, 15, unreachable_blocked),
        "start": (13, 1),
        "goal": (1, 13),
    }

    return layouts


def build_agent_scenarios() -> List[Dict[str, object]]:
    """
    The six agent scenarios as data (true grid, start, target), so the
    web UI can run and replay each one step by step.
    """
    return [
        {"name": "Normal movement",
         "description": "Open station, the agent plans and walks to the target.",
         "grid": create_weighted_grid(15, 15), "start": (13, 1), "target": (11, 3)},
        {"name": "Neighbouring cell is blocked",
         "description": "The direct route is blocked; the agent senses it and goes round.",
         "grid": create_weighted_grid(15, 15, {(12, 1)}), "start": (13, 1), "target": (11, 1)},
        {"name": "Reaching the incident location",
         "description": "The agent starts on the incident cell and raises the arrival alert.",
         "grid": create_weighted_grid(15, 15), "start": (12, 2), "target": (12, 2)},
        {"name": "No safe neighbouring move",
         "description": "All four neighbours are blocked, so the agent raises an alert.",
         "grid": create_weighted_grid(15, 15, {(4, 5), (5, 6), (6, 5), (5, 4)}),
         "start": (5, 5), "target": (1, 1)},
        {"name": "Re-planning after a hidden hazard",
         "description": "An unknown wall blocks the planned route; the agent re-plans on the way.",
         "grid": create_weighted_grid(15, 15, {(7, 9), (8, 9), (9, 9)}),
         "start": (8, 6), "target": (8, 12)},
        {"name": "Target is unreachable",
         "description": "The target is walled in; the agent explores, then raises an alert.",
         "grid": create_weighted_grid(15, 15, {(0, 1), (1, 0), (1, 2), (2, 1)}),
         "start": (6, 6), "target": (1, 1)},
    ]


def demonstrate_model_based_agent() -> None:
    """Run at least five model-based reflex-agent demonstrations."""
    print("\nModel-Based Reflex Agent Demonstrations")
    print("=" * 55)

    station_map = create_weighted_grid(15, 15)

    print("Scenario 1: Normal movement")
    grid = create_weighted_grid(15, 15)
    agent = RailwayResponseAgent(start=(13, 1), target=(11, 3), station_map=station_map)
    agent.print_status(grid)

    print("Scenario 2: Neighbouring cell is blocked")
    grid = create_weighted_grid(15, 15, {(12, 1)})
    agent = RailwayResponseAgent(start=(13, 1), target=(11, 1), station_map=station_map)
    agent.print_status(grid)

    print("Scenario 3: Reaching the incident location")
    grid = create_weighted_grid(15, 15)
    agent = RailwayResponseAgent(start=(12, 2), target=(12, 2), station_map=station_map)
    agent.print_status(grid)

    print("Scenario 4: No safe neighbouring move")
    grid = create_weighted_grid(15, 15, {(4, 5), (5, 6), (6, 5), (5, 4)})
    agent = RailwayResponseAgent(start=(5, 5), target=(1, 1), station_map=station_map)
    agent.print_status(grid)

    print("Scenario 5: Re-planning after a newly discovered hazard")
    # The agent does not know about the wall at column 9. It plans a straight
    # route, senses the block when it arrives next to it, and re-plans.
    grid = create_weighted_grid(15, 15, {(7, 9), (8, 9), (9, 9)})
    agent = RailwayResponseAgent(start=(8, 6), target=(8, 12), station_map=station_map)
    result = agent.run_episode(grid)
    print("Route taken:", agent.path_taken)
    print("Discovered blocked cells:", sorted(agent.discovered_blocked_cells))
    print("Re-plans:", result["blocked_replans"])
    print("Final action:", result["final_action"])
    print("Route cost:", result["route_cost"])
    print("Performance score:", result["performance"])
    print()

    print("Scenario 6: Alert when the target is unreachable")
    # The target (1, 1) is surrounded on all four sides. The agent only finds
    # this out by exploring, then raises an alert.
    grid = create_weighted_grid(15, 15, {(0, 1), (1, 0), (1, 2), (2, 1)})
    agent = RailwayResponseAgent(start=(6, 6), target=(1, 1), station_map=station_map)
    result = agent.run_episode(grid)
    print("Steps taken before the alert:", result["steps"])
    print("Discovered blocked cells:", sorted(agent.discovered_blocked_cells))
    print("Re-plans:", result["blocked_replans"])
    print("Final action:", result["final_action"])
    print("Route cost:", result["route_cost"])
    print("Performance score:", result["performance"])
    print()

    print("Performance = 100 * successful_targets - route_cost "
          "- 5 * blocked_replans - wait_count")


def run_layout_experiment(layout_name: str, layout: Dict[str, object]) -> None:
    """Run and display all searches for one layout."""
    grid = layout["grid"]
    start = layout["start"]
    goal = layout["goal"]

    print(f"\n{'=' * 70}")
    print(layout_name)
    print(f"{'=' * 70}")
    print("Start:", start)
    print("Destination:", goal)

    validate_positions(grid, start, goal, {
        (row, col)
        for row in range(grid.shape[0])
        for col in range(grid.shape[1])
        if grid[row, col] == -1
    })

    results = run_all_searches(grid, start, goal)
    print_comparison_table(results)

    astar_path = results["A*"]["path"]
    if astar_path is None:
        print("\nNo path exists")
    else:
        print("\nA* path:", astar_path)
        print("A* total cost:", float(results["A*"]["cost"]))

    visualise_grid(
        grid=grid,
        start=start,
        goal=goal,
        chosen_path=astar_path,
        title=f"{layout_name} - Weighted Grid and A* Path",
    )


def run_module1() -> None:
    """Run all requirements for Module 1 independently."""
    demonstrate_model_based_agent()

    layouts = create_layouts()
    for layout_name, layout in layouts.items():
        run_layout_experiment(layout_name, layout)


def demonstrate_dataset_layout() -> None:
    """Load the supplied CSV and display a weighted incident-based grid."""
    dataset = load_railway_dataset()
    grid, start, goal, incident_rows = create_layout_from_dataset(dataset)

    print("First five incident records:")
    print(dataset.head().to_string(index=False))
    print()
    print("Grid size:", grid.shape)
    print("CSV incident positions used:", len(incident_rows))
    print("Agent start:", start)
    print("Agent goal:", goal)

    results = run_all_searches(grid, start, goal)
    print_comparison_table(results)
    visualise_grid(
        grid=grid,
        start=start,
        goal=goal,
        chosen_path=results["A*"]["path"],
        title="CSV Incident-Based Railway Station Grid",
    )


def module1_menu() -> None:
    """Console menu entry point for Module 1."""
    while True:
        print("\nMODULE 1 - Agent and Search")
        print("1. Run complete Module 1")
        print("2. Demonstrate model-based agent")
        print("3. Compare searches on all layouts")
        print("4. Load supplied CSV and run incident-based grid")
        print("5. Exit Module 1")

        try:
            choice = input("Enter your choice: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nReturning to the main application.")
            break

        if choice == "1":
            run_module1()
        elif choice == "2":
            demonstrate_model_based_agent()
        elif choice == "3":
            for layout_name, layout in create_layouts().items():
                run_layout_experiment(layout_name, layout)
        elif choice == "4":
            demonstrate_dataset_layout()
        elif choice == "5":
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    module1_menu()
