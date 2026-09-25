"""Unit tests for the JSR core package using standard library unittest."""
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jsr import monitor, selector, backend


class TestJSRCore(unittest.TestCase):
    def test_monitor_risk_computation(self):
        """Verify monitor produces correct risk snapshots without future info."""
        indptr = np.array([0, 1, 2], dtype=np.int64)
        indices = np.array([0, 1], dtype=np.int64)
        data0 = np.array([2.0, 2.0], dtype=np.float64)
        data1 = np.array([2.5, 2.0], dtype=np.float64)

        mat0 = {"indptr": indptr, "indices": indices, "data": data0, "shape": (2, 2)}
        mat1 = {"indptr": indptr, "indices": indices, "data": data1, "shape": (2, 2)}

        state_matrices = {0: mat0, 1: mat1}
        domains = [np.array([0], dtype=np.int64), np.array([1], dtype=np.int64)]

        snapshot = monitor.compute_snapshot(
            current=mat1,
            domains=domains,
            local_build_states=[0, 0],
            local_ages=[1, 1],
            state_matrices=state_matrices,
            coarse_build_state=0,
            previous_state=0,
            current_state=1,
        )

        self.assertEqual(len(snapshot.local_drift), 2)
        self.assertGreater(snapshot.local_drift[0], 0.0)
        self.assertEqual(snapshot.local_drift[1], 0.0)
        self.assertFalse(snapshot.future_information_used)

    def test_selector_action_decision(self):
        """Verify selector correctly selects high-risk subdomains and decides repair action."""
        risks = [0.01] * 9 + [1.0]
        drifts = [0.01] * 9 + [1.0]
        ages = [0] * 10

        snapshot = monitor.RiskSnapshot(
            previous_state=0,
            current_state=1,
            local_drift=tuple(drifts),
            local_risk=tuple(risks),
            local_age=tuple(ages),
            coarse_matrix_risk=0.5,
            coarse_age=1,
        )

        history = selector.OnlineHistory()
        config = selector.SelectorConfig()

        required = selector.required_local_blocks(snapshot, config)
        self.assertIn(9, required)

        selected, candidates = selector.select_action(
            snapshot=snapshot,
            factor_count=10,
            history=history,
            config=config,
        )

        self.assertIsNotNone(selected)
        self.assertIn(selected.action_id, [
            "reuse", "local_partial", "joint_partial", "coarse_only", "full_rebuild"
        ])


if __name__ == "__main__":
    unittest.main()
