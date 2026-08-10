import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml
from numpy import float64 as fp64

from knife_edge_billiard.config import BilliardConfig, CoinConfig, SimConfig
from knife_edge_billiard import Billiard, Coin

REF_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "penny-coin.yaml"


# J is intentionally absent: it is derived from m and l unless overridden.
VALID_PAYLOAD = {
    "billiard": {"R": 0.2},
    "coin": {
        "x": 0.0, "y": 0.19, "theta": np.pi / 2,
        "v": 0.1, "omega": 0.2,
        "l": 0.01, "m": 0.0025,
    },
}


class BilliardConfigTests(unittest.TestCase):
    def test_builds_fp64_dataclass(self) -> None:
        billiard = BilliardConfig(R=0.2).to_billiard()
        self.assertIsInstance(billiard, Billiard)
        self.assertIsInstance(billiard.R, fp64)
        self.assertEqual(float(billiard.R), 0.2)

    def test_nonpositive_R_raises(self) -> None:
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                BilliardConfig(R=bad)

    def test_unknown_key_raises(self) -> None:
        with self.assertRaises(ValueError):
            BilliardConfig(R=0.2, bogus=1.0)

    def test_missing_R_raises(self) -> None:
        with self.assertRaises(ValueError):
            BilliardConfig()


class CoinConfigTests(unittest.TestCase):
    def test_builds_fp64_dataclass(self) -> None:
        coin = CoinConfig(**VALID_PAYLOAD["coin"]).to_coin()
        self.assertIsInstance(coin, Coin)
        self.assertIsInstance(coin.v, fp64)
        self.assertEqual(float(coin.y), 0.19)

    def test_nonpositive_physical_property_raises(self) -> None:
        for field in ("l", "m", "J"):
            payload = {**VALID_PAYLOAD["coin"], field: 0.0}
            with self.assertRaises(ValueError):
                CoinConfig(**payload)

    def test_missing_field_raises(self) -> None:
        payload = {k: v for k, v in VALID_PAYLOAD["coin"].items() if k != "l"}
        with self.assertRaises(ValueError):
            CoinConfig(**payload)

    def test_J_is_derived_when_omitted(self) -> None:
        # uniform bar spanning [-l, +l]: J = m*l^2/3 = m*(2l)^2/12
        cfg = CoinConfig(**VALID_PAYLOAD["coin"])
        m, l = cfg.m, cfg.l
        self.assertIsNone(cfg.J)
        self.assertAlmostEqual(cfg.inertia, m * l**2 / 3.0)
        self.assertAlmostEqual(cfg.inertia, m * (2 * l) ** 2 / 12.0)
        self.assertAlmostEqual(float(cfg.to_coin().J), m * l**2 / 3.0)

    def test_explicit_J_overrides_the_derived_value(self) -> None:
        # a legal non-uniform distribution: heavier towards the endpoints
        payload = {**VALID_PAYLOAD["coin"], "J": 0.5 * 0.0025 * 0.01**2}
        cfg = CoinConfig(**payload)
        self.assertAlmostEqual(cfg.inertia, 0.5 * 0.0025 * 0.01**2)
        self.assertAlmostEqual(float(cfg.to_coin().J), 0.5 * 0.0025 * 0.01**2)

    def test_J_at_the_all_mass_at_endpoints_bound_is_accepted(self) -> None:
        m, l = VALID_PAYLOAD["coin"]["m"], VALID_PAYLOAD["coin"]["l"]
        cfg = CoinConfig(**{**VALID_PAYLOAD["coin"], "J": m * l**2})
        self.assertAlmostEqual(cfg.inertia, m * l**2)

    def test_J_above_the_physical_bound_raises(self) -> None:
        m, l = VALID_PAYLOAD["coin"]["m"], VALID_PAYLOAD["coin"]["l"]
        for bad in (1.000001 * m * l**2, 10.0 * m * l**2):
            with self.assertRaises(ValueError):
                CoinConfig(**{**VALID_PAYLOAD["coin"], "J": bad})

    def test_unknown_key_raises(self) -> None:
        with self.assertRaises(ValueError):
            CoinConfig(**VALID_PAYLOAD["coin"], bogus=1.0)

    def test_theta_accepts_pi_expression_string(self) -> None:
        coin = CoinConfig(**{**VALID_PAYLOAD["coin"], "theta": "pi/2"}).to_coin()
        self.assertAlmostEqual(float(coin.theta), np.pi / 2)

    def test_theta_is_normalized_to_pi_range(self) -> None:
        # normalize maps everything into [-pi, pi]; pi itself maps to -pi
        cases = {"2*pi": 0.0, "pi": -np.pi, "-pi/2": -np.pi / 2, "pi + pi/2": -np.pi / 2}
        for expr, expected in cases.items():
            coin = CoinConfig(**{**VALID_PAYLOAD["coin"], "theta": expr}).to_coin()
            self.assertAlmostEqual(float(coin.theta), expected, msg=expr)

    def test_theta_numeric_out_of_range_is_normalized(self) -> None:
        coin = CoinConfig(**{**VALID_PAYLOAD["coin"], "theta": 2.5 * np.pi}).to_coin()
        self.assertAlmostEqual(float(coin.theta), np.pi / 2)

    def test_theta_rejects_unsafe_or_invalid_expression(self) -> None:
        for bad in ("pi**2", "foo", "__import__('os')", "pi/0", "2pi", "()"):
            with self.assertRaises(ValueError, msg=bad):
                CoinConfig(**{**VALID_PAYLOAD["coin"], "theta": bad})

    def test_t_is_not_configurable(self) -> None:
        with self.assertRaises(ValueError):
            CoinConfig(**VALID_PAYLOAD["coin"], t=0.0)

    def test_to_coin_always_starts_at_t_zero(self) -> None:
        coin = CoinConfig(**VALID_PAYLOAD["coin"]).to_coin()
        self.assertEqual(float(coin.t), 0.0)


class SimConfigTests(unittest.TestCase):
    def test_wall_touching_config_passes_and_yields_dataclasses(self) -> None:
        billiard, coin = SimConfig.model_validate(VALID_PAYLOAD).to_dataclasses()
        self.assertIsInstance(billiard, Billiard)
        self.assertIsInstance(coin, Coin)

    def test_interior_config_passes_without_wall_contact(self) -> None:
        # both endpoints strictly interior -> valid now that wall-touch is dropped
        payload = {
            "billiard": {"R": 1.0},
            "coin": {**VALID_PAYLOAD["coin"], "x": 0.0, "y": 0.0},
        }
        SimConfig.model_validate(payload)  # should not raise

    def test_coin_outside_billiard_raises(self) -> None:
        payload = {
            "billiard": {"R": 0.2},
            "coin": {**VALID_PAYLOAD["coin"], "x": 0.5, "y": 0.0},
        }
        with self.assertRaises(ValueError):
            SimConfig.model_validate(payload)

    def test_unknown_top_level_key_raises(self) -> None:
        with self.assertRaises(ValueError):
            SimConfig.model_validate({**VALID_PAYLOAD, "bogus": 1})

    def test_missing_section_raises(self) -> None:
        with self.assertRaises(ValueError):
            SimConfig.model_validate({"billiard": {"R": 0.2}})

    def test_non_dict_section_raises(self) -> None:
        with self.assertRaises(ValueError):
            SimConfig.model_validate({"billiard": {"R": 0.2}, "coin": [1, 2, 3]})

    def test_from_yaml_loads_ref_config(self) -> None:
        billiard, coin = SimConfig.from_yaml(REF_CONFIG).to_dataclasses()
        self.assertEqual(float(billiard.R), 0.2)
        self.assertIsInstance(coin, Coin)
        self.assertTrue(billiard.contains_coin(coin))

    def test_pretty_renders_the_derived_J_not_null(self) -> None:
        sim = SimConfig.model_validate(VALID_PAYLOAD)
        rendered = sim.pretty()
        self.assertNotIn("null", rendered)
        self.assertIn("J:", rendered)
        self.assertAlmostEqual(
            float(yaml.safe_load(rendered)["coin"]["J"]), sim.coin.inertia
        )

    def test_from_yaml_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            SimConfig.from_yaml("does/not/exist.yaml")

    def test_from_yaml_malformed_yaml_raises(self) -> None:
        with TemporaryDirectory() as d:
            path = Path(d) / "bad.yaml"
            path.write_text("billiard: {R: 0.2\ncoin: [unbalanced")
            with self.assertRaises(ValueError):
                SimConfig.from_yaml(path)


if __name__ == "__main__":
    unittest.main()
