import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml

from knife_edge_billiard import copy_config, make_run_dir, parse_angle


class ParseAngleTests(unittest.TestCase):
    def test_passes_through_numbers(self) -> None:
        self.assertEqual(parse_angle(0.5), 0.5)
        self.assertEqual(parse_angle(3), 3.0)

    def test_evaluates_pi_expressions(self) -> None:
        cases = {
            "pi": np.pi,
            "pi/2": np.pi / 2,
            "2*pi": 2 * np.pi,
            "-pi/4": -np.pi / 4,
            "(pi + pi) / 4": np.pi / 2,
        }
        for expr, expected in cases.items():
            self.assertAlmostEqual(parse_angle(expr), expected, msg=expr)

    def test_rejects_unknown_names_and_calls(self) -> None:
        # must reject arbitrary code rather than execute it
        for bad in (
            "foo",
            "tau",
            "__import__('os').system('echo hi')",
            "open('x')",
            "pi.__class__",
            "[1, 2]",
        ):
            with self.assertRaises(ValueError, msg=bad):
                parse_angle(bad)

    def test_rejects_disallowed_operators(self) -> None:
        for bad in ("pi ** 2", "pi % 2", "pi // 2"):
            with self.assertRaises(ValueError, msg=bad):
                parse_angle(bad)

    def test_rejects_syntax_errors_and_bad_types(self) -> None:
        for bad in ("2pi", "pi /", ""):
            with self.assertRaises(ValueError, msg=bad):
                parse_angle(bad)
        with self.assertRaises(ValueError):
            parse_angle(None)
        with self.assertRaises(ValueError):
            parse_angle(True)

    def test_division_by_zero_is_value_error(self) -> None:
        with self.assertRaises(ValueError):
            parse_angle("pi / 0")


class RunOutputTests(unittest.TestCase):
    def test_make_run_dir_creates_timestamped_dir_with_snapshots(self) -> None:
        with TemporaryDirectory() as d:
            run_dir = make_run_dir(base=d)
            self.assertTrue(run_dir.is_dir())
            self.assertTrue((run_dir / "snapshots").is_dir())
            self.assertEqual(run_dir.parent, Path(d))

    def test_copy_config_places_config_yaml_in_run_dir(self) -> None:
        with TemporaryDirectory() as d:
            src = Path(d) / "input.yaml"
            src.write_text(yaml.safe_dump({"billiard": {"R": 0.2}}))
            run_dir = make_run_dir(base=d)
            copy_config(src, run_dir)
            dest = run_dir / "config.yaml"
            self.assertTrue(dest.is_file())
            self.assertEqual(yaml.safe_load(dest.read_text()), {"billiard": {"R": 0.2}})


if __name__ == "__main__":
    unittest.main()
