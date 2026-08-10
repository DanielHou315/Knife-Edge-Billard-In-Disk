import unittest

import numpy as np
from numpy import float64 as fp64

from knife_edge_billiard import Billiard, Coin


class ContainsCoinTests(unittest.TestCase):
    def test_interior_coin_is_contained(self) -> None:
        billiard = Billiard(R=fp64(1.0))
        coin = Coin(x=fp64(0.0), y=fp64(0.0), theta=fp64(0.0), l=fp64(0.1))
        self.assertTrue(billiard.contains_coin(coin))

    def test_coin_outside_is_not_contained(self) -> None:
        billiard = Billiard(R=fp64(0.2))
        coin = Coin(x=fp64(0.5), y=fp64(0.0), theta=fp64(0.0), l=fp64(0.01))
        self.assertFalse(billiard.contains_coin(coin))

    def test_endpoint_on_wall_is_contained(self) -> None:
        # front endpoint at (0, 0.20) sits exactly on R=0.2
        billiard = Billiard(R=fp64(0.2))
        coin = Coin(x=fp64(0.0), y=fp64(0.19), theta=fp64(np.pi / 2), l=fp64(0.01))
        self.assertTrue(billiard.contains_coin(coin))


class ContainsCoinBoundaryTests(unittest.TestCase):
    """``contains_coin_boundary`` flags the 'already at an impact' configuration:
    contained, with exactly one endpoint touching the wall."""

    def test_exactly_one_endpoint_on_wall_is_boundary(self) -> None:
        # front at (0, 0.20) on the wall, back at (0, 0.18) interior
        billiard = Billiard(R=fp64(0.2))
        coin = Coin(x=fp64(0.0), y=fp64(0.19), theta=fp64(np.pi / 2), l=fp64(0.01))
        self.assertTrue(billiard.contains_coin_boundary(coin))

    def test_both_endpoints_interior_is_not_boundary(self) -> None:
        billiard = Billiard(R=fp64(1.0))
        coin = Coin(x=fp64(0.0), y=fp64(0.0), theta=fp64(0.0), l=fp64(0.1))
        self.assertFalse(billiard.contains_coin_boundary(coin))

    def test_coin_outside_is_not_boundary(self) -> None:
        billiard = Billiard(R=fp64(0.2))
        coin = Coin(x=fp64(0.5), y=fp64(0.0), theta=fp64(0.0), l=fp64(0.01))
        self.assertFalse(billiard.contains_coin_boundary(coin))

    def test_both_endpoints_on_wall_is_not_boundary(self) -> None:
        # coin spans the full diameter (l == R): both ends on the wall -> XOR false
        billiard = Billiard(R=fp64(1.0))
        coin = Coin(x=fp64(0.0), y=fp64(0.0), theta=fp64(0.0), l=fp64(1.0))
        self.assertFalse(billiard.contains_coin_boundary(coin))


if __name__ == "__main__":
    unittest.main()
