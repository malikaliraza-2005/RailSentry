"""Run from the project folder:  C:\\venvs\\ailab\\Scripts\\python.exe -m unittest discover tests"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import module1_agent_search as module1
import module2_data_ml as module2
import module4_integrated_system as module4
import user_data

GOOD = {"incident_id": "A", "grid_x": 5, "grid_y": 6, "incident_type": "Fire_Smoke",
        "crowd_density_pct": 85, "hazard_severity_1_5": 5, "equipment_health_pct": 40,
        "obstruction_level_0_3": 3, "hour_of_day": 18}
HEADER = ",".join(user_data.FIELDS)
LINE = "A,Medical,1,1,1,1,1,1,1"
NORMAL = module1.create_layouts()["Normal Layout"]


def row(**changes):
    return {**GOOD, **changes}


def run(rows, grid=None, start=(13, 1)):
    return user_data.analyse(rows, (NORMAL["grid"] if grid is None else grid).tolist(), start)


class CsvTests(unittest.TestCase):
    def test_reads_commas_semicolons_and_bom(self):
        for sep, bom in ((",", ""), (";", "\ufeff")):
            values = [str(GOOD[name]) for name in user_data.FIELDS]
            text = bom + HEADER.replace(",", sep) + "\n" + sep.join(values)
            self.assertEqual(user_data.parse_csv(text)["rows"][0]["incident_type"], "Fire_Smoke")

    def test_bad_files(self):
        for text in ("", "  ", HEADER, "a,b\n1,2", HEADER + ",grid_x\n1", HEADER + "\n" + ",".join("1" * 20)):
            with self.assertRaises(ValueError):
                user_data.parse_csv(text)

    def test_row_limit(self):
        with self.assertRaises(ValueError):
            user_data.parse_csv(HEADER + ("\n" + LINE) * (user_data.MAX_ROWS + 1))

    def test_extra_columns_are_reported(self):
        self.assertEqual(
            user_data.parse_csv(HEADER + ",risk_level\n" + LINE + ",High")["ignored_columns"], ["risk_level"])


class ValidationTests(unittest.TestCase):
    def test_bad_values(self):
        bad = [("crowd_density_pct", 190), ("grid_x", 15), ("grid_x", 1.5), ("hour_of_day", 24),
               ("hazard_severity_1_5", 0), ("grid_y", "abc"), ("grid_y", ""), ("grid_y", "inf"),
               ("incident_type", "Aliens"), ("incident_type", "")]
        for field, value in bad:
            valid, errors, _ = user_data.validate_rows([row(**{field: value})])
            self.assertEqual((valid, errors[0]["field"]), ([], field), (field, value))

    def test_all_problems_in_a_row_are_listed(self):
        _, errors, _ = user_data.validate_rows([row(grid_x=99, hour_of_day=-1)])
        self.assertEqual(len(errors), 2)

    def test_duplicate_ids(self):
        valid, errors, _ = user_data.validate_rows([row(), row()])
        self.assertEqual((len(valid), errors[0]["row"]), (1, 2))

    def test_type_spelling_is_flexible(self):
        valid, _, _ = user_data.validate_rows([row(incident_type="fire smoke")])
        self.assertEqual(valid[0]["incident_type"], "Fire_Smoke")

    def test_bad_maps(self):
        wall = NORMAL["grid"].copy()
        wall[13, 1] = -1
        for grid, start in ((wall, (13, 1)), (NORMAL["grid"], (15, 0)),
                            (NORMAL["grid"], (-1, 0)), (NORMAL["grid"][:5], (1, 1))):
            with self.assertRaises(ValueError):
                run([row()], grid, start)


class AnalysisTests(unittest.TestCase):
    def test_good_and_bad_rows_together(self):
        result = run([row(), row(incident_id="B", crowd_density_pct=190)])
        self.assertEqual((len(result["incidents"]), result["errors"][0]["row"]), (1, 2))

    def test_all_bad_rows(self):
        result = run([row(grid_x=99)])
        self.assertEqual((result["incidents"], result["plan"]), ([], None))

    def test_empty(self):
        with self.assertRaises(ValueError):
            run([])

    def test_matches_single_prediction_page(self):
        import module3_neural_network as module3
        expected = module3.predict_incident({k: v for k, v in GOOD.items() if k != "incident_id"})
        got = run([row()])["incidents"][0]
        self.assertEqual(got["risk_level"], expected["predicted_risk"])
        self.assertAlmostEqual(got["predicted_response_time"], expected["response_time_min"])
        self.assertAlmostEqual(sum(got["probabilities"].values()), 1, places=4)

    def test_blank_optional_values(self):
        result = run([row(crowd_density_pct="", equipment_health_pct=None, obstruction_level_0_3="NA")])
        self.assertEqual((len(result["incidents"]), len(result["warnings"])), (1, 1))

    def test_unreachable_wall_and_start_cell(self):
        grid = module1.create_layouts()["Unreachable-Goal Layout"]["grid"]
        result = run([row(incident_id="walled in", grid_x=13, grid_y=1),
                      row(incident_id="in wall", grid_x=3, grid_y=3),
                      row(incident_id="here", grid_x=1, grid_y=13)], grid)
        found = {i["incident_id"]: i for i in result["incidents"]}
        self.assertFalse(found["walled in"]["reachable"])
        self.assertFalse(found["in wall"]["reachable"])
        self.assertEqual(found["here"]["normal"]["travel_cost"], 0)
        self.assertEqual(result["incidents"][0]["incident_id"], "here")

    def test_risk_aware_route_avoids_predicted_risk(self):
        grid = np.ones((15, 15), dtype=int)
        for cell in ((0, 5),):
            data = pd.DataFrame([
                {"incident_id": "danger", "grid_x": 5, "grid_y": 0, "predicted_risk": 2, "predicted_response_time": 5.0},
                {"incident_id": "target", "grid_x": 10, "grid_y": 0, "predicted_risk": 0, "predicted_response_time": 5.0}])
            target = module4.plan_user_incidents(grid, (0, 0), data)["routes"]["target"]
            self.assertIn(cell, target["normal"]["path"])
            self.assertNotIn(cell, target["risk_aware"]["path"])
            self.assertEqual((target["normal"]["predicted_risk"], target["risk_aware"]["predicted_risk"]), (2, 0))

    def test_tour_is_connected_and_avoids_walls(self):
        result = run([row(incident_id=str(n), grid_x=n, grid_y=14 - n) for n in range(0, 15, 3)])
        tour = result["plan"]["tour"]
        for a, b in zip(tour, tour[1:]):
            self.assertEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1)
            self.assertNotEqual(NORMAL["grid"][tuple(b)], -1)
        reachable = [i["priority"] for i in result["incidents"] if i["reachable"]]
        self.assertEqual(reachable, list(range(1, len(reachable) + 1)))


class CleaningReportTests(unittest.TestCase):
    def test_each_value_belongs_to_its_incident(self):
        raw = pd.read_csv(module2.find_dataset())
        report = {}
        module2.handle_outliers(module2.handle_missing_values(raw), report)
        lookup = raw.set_index("incident_id")
        self.assertTrue(report)
        for column, info in report.items():
            for incident_id, value in zip(info["incident_ids"], info["values"]):
                self.assertEqual(lookup.loc[incident_id, column], value)


if __name__ == "__main__":
    unittest.main()
