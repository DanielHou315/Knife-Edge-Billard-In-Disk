import ast
import operator
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np


class SimConsistencyError(RuntimeError):
	"""Raised when computed simulation state violates an internal invariant."""


class InitialConditionError(ValueError):
	"""Raised when a requested simulation cannot start from the supplied state."""


# Math utils
def normalize_angle(angle):
	"""Normalize angle to be in [-pi, pi]."""
	return (angle + np.pi) % (2 * np.pi) - np.pi


# Binary / unary operators permitted inside an angle expression.
_ANGLE_BINOPS = {
	ast.Add: operator.add,
	ast.Sub: operator.sub,
	ast.Mult: operator.mul,
	ast.Div: operator.truediv,
}
_ANGLE_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}




def _eval_angle_node(node):
	"""Recursively evaluate the safe subset of an angle-expression AST."""
	if isinstance(node, ast.BinOp) and type(node.op) in _ANGLE_BINOPS:
		return _ANGLE_BINOPS[type(node.op)](
			_eval_angle_node(node.left), _eval_angle_node(node.right)
		)
	if isinstance(node, ast.UnaryOp) and type(node.op) in _ANGLE_UNARYOPS:
		return _ANGLE_UNARYOPS[type(node.op)](_eval_angle_node(node.operand))
	if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
		return float(node.value)
	if isinstance(node, ast.Name) and node.id == "pi":
		return float(np.pi)
	raise ValueError("unsupported element in angle expression")


def parse_angle(value):
	"""Parse an angle given as a number or a string expression in terms of ``pi``.

	Strings may use ``pi``, numeric literals, parentheses and the basic
	operators ``+ - * /`` (e.g. ``"pi/2"``, ``"-pi/4"``, ``"2*pi"``).  The
	expression is evaluated by walking a restricted AST -- never ``eval`` -- so
	arbitrary code (function calls, attribute access, names other than ``pi``)
	is rejected with ``ValueError``.
	"""
	if isinstance(value, bool):  # bool is an int subclass; reject it explicitly
		raise ValueError(f"angle must be a number or string, got bool {value!r}")
	if isinstance(value, (int, float)):
		return float(value)
	if not isinstance(value, str):
		raise ValueError(
			f"angle must be a number or string, got {type(value).__name__}"
		)
	try:
		tree = ast.parse(value, mode="eval")
		return _eval_angle_node(tree.body)
	except (SyntaxError, ValueError, TypeError, ZeroDivisionError) as exc:
		raise ValueError(f"invalid angle expression {value!r}: {exc}") from exc

# Path utils

def make_run_dir(base="data"):
    """Create data/<YYYY-MM-DD_HH-MM-SS>/ output folders and return the run directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base) / timestamp
    (run_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    (run_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
    return run_dir


def copy_config(config_path, run_dir):
    """Copy the input config YAML into the run directory as config.yaml."""
    shutil.copyfile(config_path, Path(run_dir) / "config.yaml")

# Video utils
def video_progress(current_frame, total_frames):
    print(f"\r  frame {current_frame + 1}/{total_frames}", end="", flush=True)
