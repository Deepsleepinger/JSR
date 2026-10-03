"""Mathematical contract checks for the shadow controller, without FEM imports."""
import ast
import itertools
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_controller():
    # The policy itself only needs NumPy; avoid importing the benchmark's FEM
    # dependencies so these contract checks also run in lightweight CI.
    source = Path(__file__).resolve().parents[1] / "benchmarks/run_dc_jsr_shadow_prototype.py"
    definition = next(node for node in ast.parse(source.read_text()).body
                      if isinstance(node, ast.ClassDef) and node.name == "DC_JSR_Controller")
    namespace = dict(np=np, Dict=Dict, List=List, Tuple=Tuple)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["DC_JSR_Controller"]


Controller = load_controller()


class TestDCJSRPolicy(unittest.TestCase):
    def test_max_selection_is_exhaustive_positive_cost_optimum(self):
        rng = np.random.default_rng(20261003)
        indices = [np.array([i]) for i in range(8)]
        for _ in range(200):
            values = rng.uniform(0, .2, 8)
            costs = rng.uniform(.1, 10, 8)
            controller = Controller(epsilon=.08)
            controller.V[:] = values
            _, selected, _ = controller.update_and_decide(1, np.ones(8), np.ones(8), indices)
            feasible = [subset for k in range(9) for subset in itertools.combinations(range(8), k)
                        if max((values[i] for i in range(8) if i not in subset), default=0) <= .08]
            optimum = min(sum(costs[i] for i in subset) for subset in feasible)
            self.assertAlmostEqual(sum(costs[i] for i in selected), optimum)
            self.assertEqual(len(selected), min(map(len, feasible)))

    def test_boundary_and_zero_budget(self):
        controller = Controller(n_sub=3, epsilon=.08)
        controller.V[:] = [0, .08, np.nextafter(.08, np.inf)]
        indices = [np.array([i]) for i in range(3)]
        k, selected, _ = controller.update_and_decide(1, np.ones(3), np.ones(3), indices)
        self.assertEqual((k, selected), (1, [2]))
        k, selected, _ = controller.update_and_decide(2, np.ones(3), np.ones(3), indices)
        self.assertEqual((k, selected), (0, []))

    def test_closed_path_remains_nonzero(self):
        indices = [np.array([0])]
        path = Controller(n_sub=1, epsilon=10)
        endpoint = Controller(n_sub=1, epsilon=10, metric_type="endpoint")
        for controller in (path, endpoint):
            controller.update_and_decide(1, np.array([2.]), np.array([1.]), indices)
            controller.update_and_decide(2, np.array([1.]), np.array([2.]), indices)
        self.assertEqual(path.V[0], 1.5)
        self.assertEqual(endpoint.V[0], 0)

    def test_normalized_endpoint_requires_norm_ratio(self):
        controller = Controller(n_sub=1, epsilon=100)
        indices = [np.array([0])]
        controller.update_and_decide(1, np.array([5.]), np.array([10.]), indices)
        controller.update_and_decide(2, np.array([1.]), np.array([5.]), indices)
        endpoint = 9.
        self.assertEqual(controller.V[0], 5.)
        self.assertGreater(endpoint, controller.V[0])
        self.assertLessEqual(endpoint, 5. * controller.V[0])

    def test_refresh_starts_a_new_path(self):
        controller = Controller(n_sub=1, epsilon=.08)
        indices = [np.array([0])]
        _, selected, _ = controller.update_and_decide(1, np.array([2.]), np.array([1.]), indices)
        self.assertEqual(selected, [0])
        self.assertEqual(controller.V[0], 0)
        self.assertEqual(controller.tau[0], 1)
        controller.update_and_decide(2, np.array([2.1]), np.array([2.]), indices)
        self.assertAlmostEqual(controller.V[0], .1 / 2.1)
        self.assertEqual(controller.tau[0], 1)


if __name__ == "__main__":
    unittest.main()
