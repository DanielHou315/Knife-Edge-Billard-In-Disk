import unittest
from copy import deepcopy

import numpy as np
from numpy import float64 as fp64

from knife_edge_billiard import (
    Billiard,
    Coin,
    elastic_impact,
    free_space_transition,
    first_impact,
    InitialConditionError,
    SimConsistencyError,
)
from knife_edge_billiard.data import ImpactData
from knife_edge_billiard.utils import normalize_angle


def kinetic_energy(coin: Coin) -> fp64:
    return fp64(0.5 * coin.m * coin.v**2 + 0.5 * coin.J * coin.omega**2)


def integrated_midpoint(coin: Coin, dt: fp64) -> np.ndarray:
    """Evaluate the exact free-space midpoint motion after dt."""
    k = coin.v / coin.omega
    center = np.array([
        coin.x - k * np.sin(coin.theta),
        coin.y + k * np.cos(coin.theta),
    ])
    theta = coin.theta + coin.omega * dt
    return center + np.array([
        k * np.sin(theta),
        -k * np.cos(theta),
    ])


def random_incoming_coin(rng: np.random.Generator) -> Coin:
    """Construct a forward-moving state with the front endpoint on the wall.

    The impact map assumes the contact point is the front endpoint
    (``sign(v) == +1``), so configurations are restricted to ``v > 0``.
    """
    phi = rng.uniform(-np.pi, np.pi)
    theta = rng.uniform(-np.pi, np.pi)
    length = rng.uniform(0.03, 0.25)
    sigma = 1.0
    v = rng.uniform(0.1, 2.0)
    omega = rng.choice((-1.0, 1.0)) * rng.uniform(0.1, 3.0)

    wall_point = np.array([np.cos(phi), np.sin(phi)])
    direction = np.array([np.cos(theta), np.sin(theta)])
    midpoint = wall_point - sigma * length * direction

    return Coin(
        x=fp64(midpoint[0]),
        y=fp64(midpoint[1]),
        theta=fp64(theta),
        v=fp64(v),
        omega=fp64(omega),
        l=fp64(length),
        m=fp64(rng.uniform(0.5, 2.0)),
        J=fp64(rng.uniform(0.03, 0.4)),
        t=fp64(0.0),
    )


def has_outward_contact_velocity(coin: Coin) -> bool:
    """Return whether the selected endpoint is moving out through the wall."""
    sigma = np.sign(coin.v)
    direction = np.array([np.cos(coin.theta), np.sin(coin.theta)])
    tangent = np.array([-np.sin(coin.theta), np.cos(coin.theta)])
    endpoint = coin.pos + sigma * coin.l * direction
    endpoint_velocity = coin.v * direction + sigma * coin.l * coin.omega * tangent
    return bool(endpoint @ endpoint_velocity > 1e-8)


def endpoint_circle_reaches_wall(coin: Coin, billiard: Billiard) -> bool:
    """Whether the endpoint circle transversally crosses the wall (i.e. an impact
    exists). False for a fully-inscribed or otherwise non-intersecting circle."""
    k = coin.v / coin.omega
    r = float(np.hypot(k, coin.l))
    center = np.array([coin.x - k * np.sin(coin.theta), coin.y + k * np.cos(coin.theta)])
    d = float(np.linalg.norm(center))
    return d > 0.0 and abs(billiard.R - r) <= d <= billiard.R + r


class RandomTransitionTests(unittest.TestCase):
    def test_singular_v_zero_pure_rotation(self) -> None:
        """Singular case v=0 (Section 3.3): the midpoint is at rest and the body
        rotates. first_impact must land the earliest endpoint on the wall, the
        impact must conserve energy, and (since a contained v=0 impact always
        produces v!=0) the flow continues into the generic branch."""
        rng = np.random.default_rng(31)
        billiard = Billiard(R=fp64(1.0))
        checked = 0
        for _ in range(400):
            d = rng.uniform(0.3, 0.95)
            psi = rng.uniform(-np.pi, np.pi)
            coin = Coin(
                x=fp64(d * np.cos(psi)), y=fp64(d * np.sin(psi)),
                theta=fp64(rng.uniform(-np.pi, np.pi)),
                v=fp64(0.0), omega=fp64(rng.choice((-1.0, 1.0)) * rng.uniform(0.3, 3.0)),
                l=fp64(rng.uniform(0.05, 0.3)), m=fp64(1.0), J=fp64(0.1),
            )
            if not billiard.contains_coin(coin):
                continue
            if not endpoint_circle_reaches_wall(coin, billiard):
                continue

            coin_seq, _ = first_impact(coin, billiard)
            last = coin_seq[-1]
            # a valid impact configuration, and no endpoint left the disk earlier
            self.assertTrue(billiard.contains_coin_boundary(last, tol=1e-7))
            for t in np.linspace(0.0, last.t, 200)[:-1]:
                th = coin.theta + coin.omega * t
                for s in (1.0, -1.0):
                    ep = coin.pos + s * coin.l * np.array([np.cos(th), np.sin(th)])
                    self.assertLessEqual(np.linalg.norm(ep), billiard.R + 1e-7)

            # v=0 impact: energy conserved and the state leaves the v=0 set
            post, impact = elastic_impact(billiard, last)
            self.assertAlmostEqual(kinetic_energy(last), kinetic_energy(post))
            self.assertNotAlmostEqual(post.v, 0.0)
            # generic free flight from the post-impact state stays inside
            nxt_seq, _ = free_space_transition(post, billiard, impact)
            self.assertTrue(billiard.contains_coin_boundary(nxt_seq[-1], tol=1e-7))
            checked += 1

        self.assertGreater(checked, 30)

    def test_singular_v_zero_free_space_transition(self) -> None:
        """v=0 post-impact states (Section 3.3): the midpoint stays fixed, both
        endpoints trace the circle of radius l about it (r=l, gamma=pi/2), and
        the OPPOSITE endpoint impacts next after sweeping 2*beta - pi, at wall
        angle phi - 2*sgn(omega)*alpha."""
        rng = np.random.default_rng(57)
        billiard = Billiard(R=fp64(1.0))
        checked = 0
        for _ in range(300):
            length = rng.uniform(0.05, 0.4)
            # K crosses the wall (d > R-l) and the far endpoint fits (d^2+l^2<R^2)
            d = rng.uniform(1.0 - length + 1e-6, np.sqrt(1.0 - length**2) - 1e-6)
            psi = rng.uniform(-np.pi, np.pi)
            omega = rng.choice((-1.0, 1.0)) * rng.uniform(0.3, 3.0)
            sigma_last = rng.choice((1.0, -1.0))
            omgs = np.sign(omega)

            alpha = np.arccos((d**2 + 1.0 - length**2) / (2 * d))
            beta = np.arccos((d**2 - 1.0 + length**2) / (2 * d * length))
            # contact endpoint at the entering crossing of K with the wall
            mu_s = psi + np.pi - omgs * beta
            theta = mu_s if sigma_last > 0 else mu_s - np.pi
            phi_last = normalize_angle(psi + omgs * alpha)
            coin = Coin(
                x=fp64(d * np.cos(psi)), y=fp64(d * np.sin(psi)),
                theta=fp64(normalize_angle(theta)),
                v=fp64(0.0), omega=fp64(omega),
                l=fp64(length), m=fp64(1.0), J=fp64(0.1),
            )
            if not billiard.contains_coin_boundary(coin, tol=1e-7):
                continue

            impact = ImpactData(
                impact_pt=np.array([np.cos(phi_last), np.sin(phi_last)]),
                t=fp64(0.0), phi=fp64(phi_last),
                is_leading_impact=bool(rng.choice((True, False))),
            )
            coin_seq, arc = free_space_transition(coin, billiard, impact)
            nxt = coin_seq[-1]
            checked += 1

            # midpoint is at rest; swept angle and wall angle follow the map
            np.testing.assert_allclose(nxt.pos, coin.pos, atol=1e-9)
            swept = 2 * beta - np.pi
            self.assertAlmostEqual(nxt.t, swept / abs(omega))
            self.assertAlmostEqual(
                normalize_angle(nxt.theta - coin.theta - omgs * swept), 0.0
            )
            self.assertAlmostEqual(
                normalize_angle(arc.phi_next - (phi_last - 2 * omgs * alpha)), 0.0
            )
            # the opposite endpoint lands on the wall
            landed = nxt.pos_front if sigma_last < 0 else nxt.pos_back
            self.assertAlmostEqual(float(np.linalg.norm(landed)), 1.0)
            self.assertTrue(billiard.contains_coin_boundary(nxt, tol=1e-7))
            # both endpoints stay inside throughout the sweep
            for t in np.linspace(0.0, nxt.t, 300)[1:-1]:
                th = coin.theta + omega * t
                for s in (1.0, -1.0):
                    ep = coin.pos + s * length * np.array([np.cos(th), np.sin(th)])
                    self.assertLessEqual(np.linalg.norm(ep), 1.0 + 1e-7)
            # the recovered state is a genuine pre-impact state
            post, _ = elastic_impact(billiard, nxt)
            self.assertAlmostEqual(kinetic_energy(nxt), kinetic_energy(post))

        self.assertGreater(checked, 200)

    def test_singular_omega_zero_free_space_transition(self) -> None:
        """omega=0 post-impact states (Section 3.2): straight-line flight with
        the trailing endpoint leaving the wall. The next impact is at the
        leading endpoint, with phi' = -phi + 2*theta - sgn(v)*pi (Lemma 8) and
        Delta t = 2(-sgn(v) R cos(phi-theta) - l)/|v| (Prop 9, sign-corrected)."""
        rng = np.random.default_rng(83)
        billiard = Billiard(R=fp64(1.0))
        checked = 0
        for _ in range(400):
            length = rng.uniform(0.05, 0.4)
            v = rng.choice((-1.0, 1.0)) * rng.uniform(0.2, 2.0)
            sgv = np.sign(v)
            theta = rng.uniform(-np.pi, np.pi)
            phi = rng.uniform(-np.pi, np.pi)
            # post-impact (3.23) and containment of the leading endpoint (3.10)
            if not (length < -sgv * np.cos(phi - theta)):
                continue
            trailing = np.array([np.cos(phi), np.sin(phi)])
            mid = trailing + sgv * length * np.array([np.cos(theta), np.sin(theta)])
            coin = Coin(
                x=fp64(mid[0]), y=fp64(mid[1]), theta=fp64(theta),
                v=fp64(v), omega=fp64(0.0),
                l=fp64(length), m=fp64(1.0), J=fp64(0.1),
            )
            if not billiard.contains_coin_boundary(coin, tol=1e-7):
                continue

            impact = ImpactData(
                impact_pt=trailing, t=fp64(0.0), phi=fp64(phi),
                is_leading_impact=False,
            )
            coin_seq, arc = free_space_transition(coin, billiard, impact)
            nxt = coin_seq[-1]
            checked += 1

            dt = 2 * (-sgv * np.cos(phi - theta) - length) / abs(v)
            self.assertAlmostEqual(nxt.t, dt)
            # straight-line midpoint motion, constant heading
            np.testing.assert_allclose(
                nxt.pos,
                mid + v * dt * np.array([np.cos(theta), np.sin(theta)]),
                atol=1e-9,
            )
            self.assertAlmostEqual(normalize_angle(nxt.theta - theta), 0.0)
            # Lemma 8 wall angle, and the leading endpoint lands on the wall
            self.assertAlmostEqual(
                normalize_angle(arc.phi_next - (-phi + 2 * theta - sgv * np.pi)), 0.0
            )
            landed = nxt.pos_front if sgv > 0 else nxt.pos_back
            self.assertAlmostEqual(float(np.linalg.norm(landed)), 1.0)
            self.assertTrue(billiard.contains_coin_boundary(nxt, tol=1e-7))
            # the coin stays inside throughout the flight
            for t in np.linspace(0.0, dt, 200)[1:-1]:
                for s in (1.0, -1.0):
                    ep = mid + (v * t + s * length) * np.array([np.cos(theta), np.sin(theta)])
                    self.assertLessEqual(np.linalg.norm(ep), 1.0 + 1e-7)
            # the recovered state is a genuine pre-impact state
            post, _ = elastic_impact(billiard, nxt)
            self.assertAlmostEqual(kinetic_energy(nxt), kinetic_energy(post))

        self.assertGreater(checked, 100)

    def test_singular_omega_zero_first_impact(self) -> None:
        """omega=0 interior starts (Section 3.2): straight-line motion must land
        the leading endpoint on the wall, stay inside beforehand, and hand a
        valid pre-impact state to elastic_impact."""
        rng = np.random.default_rng(101)
        billiard = Billiard(R=fp64(1.0))
        checked = 0
        for _ in range(300):
            coin = Coin(
                x=fp64(rng.uniform(-0.6, 0.6)), y=fp64(rng.uniform(-0.6, 0.6)),
                theta=fp64(rng.uniform(-np.pi, np.pi)),
                v=fp64(rng.choice((-1.0, 1.0)) * rng.uniform(0.2, 2.0)),
                omega=fp64(0.0),
                l=fp64(rng.uniform(0.05, 0.3)),
                m=fp64(rng.uniform(0.5, 2.0)), J=fp64(rng.uniform(0.05, 0.4)),
            )
            if not billiard.contains_coin(coin):
                continue

            coin_seq, _ = first_impact(coin, billiard)
            last = coin_seq[-1]
            checked += 1

            self.assertGreater(last.t, 0.0)
            self.assertTrue(billiard.contains_coin_boundary(last, tol=1e-7))
            # heading is unchanged and the leading endpoint is the contact
            self.assertAlmostEqual(normalize_angle(last.theta - coin.theta), 0.0)
            landed = last.pos_front if coin.v > 0 else last.pos_back
            self.assertAlmostEqual(float(np.linalg.norm(landed)), 1.0)
            # straight-line midpoint motion; inside the disk along the way
            e_dir = np.array([np.cos(coin.theta), np.sin(coin.theta)])
            np.testing.assert_allclose(
                last.pos, coin.pos + coin.v * last.t * e_dir, atol=1e-9
            )
            for t in np.linspace(0.0, float(last.t), 200)[:-1]:
                for s in (1.0, -1.0):
                    ep = coin.pos + (coin.v * t + s * coin.l) * e_dir
                    self.assertLessEqual(np.linalg.norm(ep), 1.0 + 1e-7)
            # admissible pre-impact state, elastic impact conserves energy
            post, _ = elastic_impact(billiard, last)
            self.assertAlmostEqual(kinetic_energy(last), kinetic_energy(post))

        self.assertGreater(checked, 100)

    def test_inward_and_tangent_contacts_are_rejected(self) -> None:
        """elastic_impact must reject states whose contact point is not moving
        strictly outward (Remark 1 admissibility): inward (dh<0) and tangent
        (dh=0). Outward states with the same geometry are accepted."""
        billiard = Billiard(R=fp64(1.0))
        length = 0.2
        # front endpoint at (1, 0) on the wall, body axis along +x
        coin = Coin(
            x=fp64(1.0 - length), y=fp64(0.0), theta=fp64(0.0),
            v=fp64(0.0), omega=fp64(0.0), l=fp64(length), m=fp64(1.0), J=fp64(0.1),
        )

        # inward: v < 0 pushes the contact point back into the disk
        inward = deepcopy(coin); inward.v = fp64(-0.5); inward.omega = fp64(0.1)
        with self.assertRaises(InitialConditionError):
            elastic_impact(billiard, inward)

        # tangent: velocity purely along the wall (dh == 0 here, phi == theta)
        tangent = deepcopy(coin); tangent.v = fp64(0.0); tangent.omega = fp64(1.0)
        with self.assertRaises(InitialConditionError):
            elastic_impact(billiard, tangent)

        # outward: same geometry, v > 0 -> accepted, energy conserved
        outward = deepcopy(coin); outward.v = fp64(0.5); outward.omega = fp64(0.1)
        post, _ = elastic_impact(billiard, outward)
        self.assertAlmostEqual(kinetic_energy(outward), kinetic_energy(post))

    def test_random_accepted_transitions_match_free_space_dynamics(self) -> None:
        rng = np.random.default_rng(4)
        billiard = Billiard(R=fp64(1.0))
        valid_count = 0
        accepted_count = 0
        rejected_count = 0

        for _ in range(5000):
            coin = random_incoming_coin(rng)
            if not billiard.contains_coin(coin):
                continue
            if not has_outward_contact_velocity(coin):
                continue

            valid_count += 1
            before_energy = kinetic_energy(coin)
            coin_after_impact, impact = elastic_impact(billiard, coin)
            self.assertAlmostEqual(before_energy, kinetic_energy(coin_after_impact))

            try:
                coin_seq, _ = free_space_transition(
                    coin_after_impact,
                    billiard,
                    impact,
                )
                next_coin = coin_seq[-1]
            except AssertionError:
                # Known limitation: the opposite endpoint can cross first.
                rejected_count += 1
                continue

            dt = next_coin.t - coin_after_impact.t
            self.assertGreater(dt, 0.0)
            np.testing.assert_allclose(
                next_coin.pos,
                integrated_midpoint(coin_after_impact, dt),
                atol=1e-12,
            )
            self.assertTrue(billiard.contains_coin(next_coin))
            accepted_count += 1

        self.assertEqual(valid_count, 2129)
        self.assertEqual(accepted_count, valid_count)
        self.assertEqual(rejected_count, 0)

    def test_leading_and_trailing_branches_are_valid(self) -> None:
        """Chained impacts must conserve energy and stay a valid billiard
        trajectory through BOTH the leading-type (rho_l) and trailing-type
        (rho_T) rotational-map branches.

        The leading branch (post-impact state whose leading endpoint rests on
        the wall, so the trailing endpoint impacts next) only arises after an
        impact that does not flip v; a single forward impact is not enough to
        reach it, so we chain many impacts per start state.
        """
        rng = np.random.default_rng(123)
        billiard = Billiard(R=fp64(1.0))
        leading_seen = 0
        trailing_seen = 0
        transitions_checked = 0

        for _ in range(300):
            coin = random_incoming_coin(rng)
            if not billiard.contains_coin(coin):
                continue
            if not has_outward_contact_velocity(coin):
                continue

            energy0 = kinetic_energy(coin)
            pre_impact = coin
            for _step in range(20):
                post, impact = elastic_impact(billiard, pre_impact)
                # Elastic impacts conserve kinetic energy regardless of branch.
                self.assertAlmostEqual(energy0, kinetic_energy(post))
                if impact.is_leading_impact:
                    leading_seen += 1
                else:
                    trailing_seen += 1

                try:
                    coin_seq, _ = free_space_transition(post, billiard, impact)
                except (SimConsistencyError, InitialConditionError):
                    # Geometric dead-ends (e.g. the opposite endpoint would
                    # cross first) legitimately end a chain.
                    break

                next_coin = coin_seq[-1]
                # The recovered state must be a genuine impact configuration
                # and the whole free-flight arc must remain inside the disk.
                self.assertTrue(billiard.contains_coin_boundary(next_coin, tol=1e-7))
                for c in coin_seq:
                    self.assertTrue(billiard.contains_coin(c, tol=1e-6))
                # Free flight preserves the quasi-velocities.
                self.assertAlmostEqual(next_coin.v, post.v)
                self.assertAlmostEqual(next_coin.omega, post.omega)
                self.assertAlmostEqual(energy0, kinetic_energy(next_coin))
                transitions_checked += 1
                pre_impact = next_coin

        # Both branches must actually be exercised for this test to be meaningful.
        self.assertGreater(leading_seen, 0, "leading-type branch never exercised")
        self.assertGreater(trailing_seen, 0, "trailing-type branch never exercised")
        self.assertGreater(transitions_checked, 100)

    def test_first_impact_lands_endpoint_on_wall(self) -> None:
        """first_impact must place the impacting endpoint exactly on the wall at
        the computed time, following the exact free-space motion, from a generic
        interior start."""
        rng = np.random.default_rng(2024)
        billiard = Billiard(R=fp64(1.0))
        checked = 0

        for _ in range(500):
            coin = Coin(
                x=fp64(rng.uniform(-0.6, 0.6)), y=fp64(rng.uniform(-0.6, 0.6)),
                theta=fp64(rng.uniform(-np.pi, np.pi)),
                v=fp64(rng.choice((-1.0, 1.0)) * rng.uniform(0.3, 2.0)),
                omega=fp64(rng.choice((-1.0, 1.0)) * rng.uniform(0.3, 3.0)),
                l=fp64(rng.uniform(0.05, 0.3)),
                m=fp64(rng.uniform(0.5, 2.0)), J=fp64(rng.uniform(0.05, 0.4)),
            )
            if not billiard.contains_coin(coin):
                continue
            # Discard configs whose endpoint circle never reaches the wall
            # (fully inscribed / non-intersecting) -- they have no first impact.
            if not endpoint_circle_reaches_wall(coin, billiard):
                continue

            coin_seq, _ = first_impact(coin, billiard)
            last = coin_seq[-1]
            self.assertGreater(last.t, 0.0)
            # exactly one endpoint on the wall, coin otherwise inside
            self.assertTrue(billiard.contains_coin_boundary(last, tol=1e-7))
            for c in coin_seq:
                self.assertTrue(billiard.contains_coin(c, tol=1e-6))
            # the impact time reproduces the exact free-space midpoint motion
            np.testing.assert_allclose(
                last.pos, integrated_midpoint(coin, last.t), atol=1e-9
            )
            checked += 1

        self.assertGreater(checked, 100)

    def test_previously_rejected_transition(self) -> None:
        billiard = Billiard(R=fp64(1.0))
        coin = Coin(
            x=fp64(0.2559895132451875),
            y=fp64(-0.9337778649740123),
            theta=fp64(2.4247498734901836),
            v=fp64(-0.19458036826795233),
            omega=fp64(2.700890610827219),
            l=fp64(0.03787645871904379),
            m=fp64(1.6433790334021317),
            J=fp64(0.3996145461894265),
            t=fp64(0.0),
        )

        coin_after_impact, impact = elastic_impact(billiard, coin)
        coin_seq, _ = free_space_transition(coin_after_impact, billiard, impact)
        self.assertTrue(billiard.contains_coin(coin_seq[-1]))


if __name__ == "__main__":
    unittest.main()
