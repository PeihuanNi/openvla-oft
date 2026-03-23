"""Utilities for Action-aware Dynamic Pruning (ADP) in LIBERO evaluation."""

import json
import os
from typing import Any, Dict, List, Sequence

import numpy as np


def load_adp_config_from_json(cfg: Any) -> None:
    """Load optional ADP/QK settings from a JSON file into a config object."""
    config_path = getattr(cfg, "qk_config_json", None)
    if not config_path or not os.path.isfile(config_path):
        return

    with open(config_path, "r") as file:
        data = json.load(file)

    for key, value in data.items():
        if not hasattr(cfg, key):
            continue
        if key == "qk_keep_split" and isinstance(value, list):
            value = ",".join(str(float(x)) for x in value)
        setattr(cfg, key, value)


class ActionAwarePruningController:
    """Window-level controller that switches between full and pruned visual inference."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.window_index = 0
        self.current_state = 0
        self.next_state = int(getattr(cfg, "initial_state", 0))
        self.delta_history: List[float] = []
        self.consecutive_pruned = 0

    def start_window(self) -> int:
        """Return the visual mode for the next action chunk."""
        cold_start_windows = max(0, int(getattr(self.cfg, "cold_start_windows", 2)))
        if self.window_index < cold_start_windows:
            self.current_state = 0
        else:
            self.current_state = int(self.next_state)
        return self.current_state

    def finish_window(self, action_chunk: Sequence[np.ndarray]) -> Dict[str, Any]:
        """Update controller state after executing one action chunk."""
        delta = self._compute_window_delta_from_actions(action_chunk)
        self.delta_history.append(delta)

        if self.current_state == 1:
            self.consecutive_pruned += 1
        else:
            self.consecutive_pruned = 0

        next_state = self._decide_next_state(delta)
        next_window_index = self.window_index + 1
        cold_start_windows = max(0, int(getattr(self.cfg, "cold_start_windows", 2)))
        if next_window_index < cold_start_windows:
            next_state = 0

        forced_refresh = False
        if bool(getattr(self.cfg, "limit_consecutive_pruned_enabled", True)):
            max_consecutive = max(1, int(getattr(self.cfg, "limit_max_consecutive_pruned", 3)))
            if self.consecutive_pruned >= max_consecutive:
                next_state = 0
                self.consecutive_pruned = 0
                forced_refresh = True

        self.next_state = int(next_state)
        self.window_index += 1

        return {
            "delta": float(delta),
            "current_state": int(self.current_state),
            "next_state": int(self.next_state),
            "forced_refresh": bool(forced_refresh),
        }

    def _decide_next_state(self, delta: float) -> int:
        decision_method = str(getattr(self.cfg, "decision_method", "adjacent")).lower()
        if decision_method == "avg":
            avg_delta = float(sum(self.delta_history) / max(1, len(self.delta_history)))
            upper = avg_delta + float(getattr(self.cfg, "hysteresis_up", 0.0))
            lower = avg_delta - float(getattr(self.cfg, "hysteresis_down", 0.0))
            if self._is_greater_equal(delta, upper):
                return 1
            if self._is_less_equal(delta, lower):
                return 0
            return int(self.current_state)

        lookback = max(1, int(getattr(self.cfg, "adjacent_lookback", 2)))
        extrema_window = max(2, int(getattr(self.cfg, "adjacent_extrema_window", 3)))
        lookback = max(lookback, extrema_window - 1)
        previous_deltas = self.delta_history[:-1]
        if not previous_deltas:
            return int(getattr(self.cfg, "initial_state", 0))

        reference = previous_deltas[-lookback:]
        upper = max(reference) + float(getattr(self.cfg, "hysteresis_up", 0.0))
        lower = min(reference) - float(getattr(self.cfg, "hysteresis_down", 0.0))
        if self._is_greater_equal(delta, upper):
            return 1
        if self._is_less_equal(delta, lower):
            return 0
        if bool(getattr(self.cfg, "adjacent_last_state", True)):
            return int(self.current_state)
        return int(getattr(self.cfg, "initial_state", 0))

    def _compute_window_delta_from_actions(self, action_chunk: Sequence[np.ndarray]) -> float:
        if not action_chunk:
            return 0.0

        positions = [np.zeros(3, dtype=np.float64)]
        rotations = [np.eye(3, dtype=np.float64)]
        position = positions[0].copy()
        rotation = rotations[0].copy()

        min_delta_pos = float(getattr(self.cfg, "min_delta_pos", 0.0))
        min_delta_rot = float(getattr(self.cfg, "min_delta_rot", 0.0))
        for action in action_chunk:
            action = np.asarray(action, dtype=np.float64).reshape(-1)
            if action.shape[0] < 6:
                continue

            translation = action[:3]
            rotation_delta = action[3:6]
            if np.linalg.norm(translation) < min_delta_pos:
                translation = np.zeros(3, dtype=np.float64)
            if np.linalg.norm(rotation_delta) < min_delta_rot:
                rotation_delta = np.zeros(3, dtype=np.float64)

            rotation_step = self._rotation_matrix_xyz(rotation_delta)
            position = position + rotation @ (rotation_step @ translation)
            rotation = rotation @ rotation_step
            positions.append(position.copy())
            rotations.append(rotation.copy())

        if len(positions) <= 1:
            return 0.0

        delta_method = str(getattr(self.cfg, "delta_method", "net")).lower()
        if delta_method in {"sum", "arc_sum"}:
            delta_pos = float(
                sum(np.linalg.norm(positions[idx] - positions[idx - 1]) for idx in range(1, len(positions)))
            )
        else:
            delta_pos = float(np.linalg.norm(positions[-1] - positions[0]))

        if "rot" not in delta_method and "combined" not in delta_method:
            return delta_pos

        delta_rot = self._relative_rotation_angle(rotations[0], rotations[-1])
        return float(delta_pos + float(getattr(self.cfg, "L_eff", 0.15)) * delta_rot)

    def _is_greater_equal(self, lhs: float, rhs: float) -> bool:
        return lhs > rhs or abs(lhs - rhs) <= float(getattr(self.cfg, "tol_equal", 0.0))

    def _is_less_equal(self, lhs: float, rhs: float) -> bool:
        return lhs < rhs or abs(lhs - rhs) <= float(getattr(self.cfg, "tol_equal", 0.0))

    @staticmethod
    def _rotation_matrix_xyz(rotation_delta: np.ndarray) -> np.ndarray:
        rx, ry, rz = rotation_delta.tolist()
        cx, cy, cz = np.cos([rx, ry, rz])
        sx, sy, sz = np.sin([rx, ry, rz])

        rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
        rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
        rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return rot_x @ rot_y @ rot_z

    @staticmethod
    def _relative_rotation_angle(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
        rotation_rel = rotation_a.T @ rotation_b
        trace = float(np.clip((np.trace(rotation_rel) - 1.0) / 2.0, -1.0, 1.0))
        angle = float(np.arccos(trace))
        return 0.0 if abs(angle) < 1e-12 else angle
