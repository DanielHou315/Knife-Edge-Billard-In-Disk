import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numpy import float64 as fp64

from knife_edge_billiard import (
    Billiard,
    Coin,
    elastic_impact,
    extract_series,
    free_space_transition,
    save_all_plots,
)

EXPECTED_PLOTS = {
    "energy_vs_impact.png",
    "arc_radius_vs_impact.png",
    "v_vs_impact.png",
    "omega_vs_impact.png",
    "v_omega_phase.png",
    "phi_vs_impact.png",
    "theta_vs_impact.png",
    "alpha_vs_impact.png",
    "beta_vs_impact.png",
    "d_vs_impact.png",
    "dt_between_impacts.png",
}


def _short_traj(billiard: Billiard, coin: Coin, num_impacts: int = 3, fps: int = 30):
    traj = []
    before = coin
    for _ in range(num_impacts + 1):
        after, impact = elastic_impact(billiard, before)
        coin_seq, arc = free_space_transition(after, billiard, impact, 1 / fps)
        traj.append({"impact_data": impact, "free_space_data": arc, "coin_seq": coin_seq})
        before = coin_seq[-1]
    return traj


class PlotsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.billiard = Billiard(R=fp64(0.2))
        self.coin = Coin(x=fp64(0.0), y=fp64(0.19), theta=fp64(np.pi / 2),
                         v=fp64(0.1), omega=fp64(0.2), l=fp64(0.01),
                         m=fp64(0.0025), J=fp64(6.25e-8))
        self.traj = _short_traj(self.billiard, self.coin)

    def test_extract_series_has_expected_keys_and_length(self) -> None:
        s = extract_series(self.traj, self.coin)
        n = len(self.traj)
        for key in ("r", "v", "omega", "theta", "phi", "d", "alpha", "beta",
                    "energy_pre", "energy_post", "impact", "t"):
            self.assertEqual(len(s[key]), n, key)
        self.assertEqual(len(s["dt"]), n - 1)

    def test_save_all_plots_writes_all_figures(self) -> None:
        with TemporaryDirectory() as d:
            save_all_plots(self.traj, self.billiard, self.coin, d)
            written = {p.name for p in Path(d).glob("*.png")}
        self.assertEqual(written, EXPECTED_PLOTS)


if __name__ == "__main__":
    unittest.main()
