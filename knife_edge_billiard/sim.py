
from copy import deepcopy
from typing import Tuple, List, Optional

import numpy as np
from numpy import float64 as fp64, ndarray

from .data import (
    Billiard, Coin,
    ImpactData,
    FreeSpaceArcData,
)
from .utils import InitialConditionError, SimConsistencyError, normalize_angle


# Implementation detail: for snapping post-impact quasi-velocities to zero when they fall below a scale-dependent threshold.
# Otherwise zero-vel path almost never triggers, and certain quantities explode past numerical limits during computation.
_ZERO_SNAP_REL: fp64 = fp64(1024) * np.finfo(np.float64).eps


def _snap_zero(value: fp64, scale: fp64) -> fp64:
    """Snap ``value`` to exact 0.0 when it is within the fp64 rounding floor of
    the given (non-vanishing) velocity ``scale`` -- see the note above."""
    return fp64(0.0) if abs(value) <= _ZERO_SNAP_REL * abs(scale) else value


def _impact_polar2cartesian(contact_sign, theta, phi, l, R) -> np.ndarray:
    """Midpoint from wall contact point; contact endpoint is p + contact_sign*l*e(theta)."""
    x: fp64 = R * np.cos(phi) - contact_sign * l * np.cos(theta)
    y: fp64 = R * np.sin(phi) - contact_sign * l * np.sin(theta)
    return np.array([x, y], dtype=np.float64)


def elastic_impact(billiard: Billiard, coin: Coin, tol=1e-9) -> Tuple[Coin, ImpactData]:
    """
    Given an immediate-before-impact coin configuration, compute the immediate-after-impact state
    """
    if not billiard.contains_coin(coin):
        raise InitialConditionError("Initial coin configuration must be inside billiard.")

    # 0. Contact endpoint sign: the endpoint on the wall (+1 front, -1 back).
    front_r = float(np.linalg.norm(coin.pos_front))
    back_r = float(np.linalg.norm(coin.pos_back))
    sigma = fp64(1.0) if abs(front_r - billiard.R) <= abs(back_r - billiard.R) else fp64(-1.0)

    # 1. Compute impact point
    impact_pt_i: ndarray = coin.pos + sigma * coin.l * \
        np.array([np.cos(coin.theta), np.sin(coin.theta)], dtype=np.float64)

    # 2. Compute Phi
    phi_i = np.atan2(
        coin.y + sigma * coin.l * np.sin(coin.theta),
        coin.x + sigma * coin.l * np.cos(coin.theta)
    )

    # 3. compute next v, omega
    # 3.1 compute the constant C
    cos_pt: fp64= np.cos(phi_i - coin.theta)
    sin_pt: fp64 = np.sin(phi_i - coin.theta)

    # Admissibility test
    dh = billiard.R * (coin.v * cos_pt + sigma * coin.l * coin.omega * sin_pt)
    if dh <= tol:
        raise InitialConditionError(
            f"Contact point is not moving outward (dh={float(dh)}); not a pre-impact state."
        )

    _C = (coin.v * cos_pt + sigma * coin.l * coin.omega * sin_pt) / \
        (cos_pt**2 / coin.m + coin.l**2 * sin_pt**2 / coin.J)

    # 3.2 actually compute next v, omega. impulse_{v,omega} are the maximum
    # magnitudes the collision could impart (|cos|,|sin| = 1); measuring the
    # result against (incoming + max impulse) lets a component killed by the
    # impact be snapped to exactly 0.0, engaging the singular omega=0 / v=0 maps
    # below, without a scale that can vanish spuriously (see _snap_zero).
    impulse_v: fp64 = 2 * _C / coin.m
    impulse_omega: fp64 = 2 * _C * coin.l / coin.J
    v_next: fp64 = _snap_zero(
        coin.v - impulse_v * cos_pt, abs(coin.v) + abs(impulse_v)
    )
    omega_next: fp64 = _snap_zero(
        coin.omega - impulse_omega * sigma * sin_pt, abs(coin.omega) + abs(impulse_omega)
    )

    # 4. Formulate impact data (leading type <=> contact endpoint is the leading one)
    impact_data = ImpactData(
        impact_pt=impact_pt_i,
        t = coin.t,
        phi=normalize_angle(phi_i),
        is_leading_impact = bool(sigma == np.sign(v_next))
    )

    # 5. Formulate next coin configuration
    coin_after_impact = deepcopy(coin)
    coin_after_impact.v = v_next
    coin_after_impact.omega = omega_next

    return coin_after_impact, impact_data


def _free_space_params(coin: Coin, b: Billiard):
    """
    Helper method to compute some shared free-space geometry parameters.
    """
    if coin.omega == 0.0:
        raise InitialConditionError("Cannot compute free space transition for zero angular velocity.")

    # 1. compute r_i
    r_i: fp64 = np.sqrt((coin.v / coin.omega)**2 + coin.l**2)

    # 2. compute d_i
    _arc_center: ndarray = np.array([
        coin.x - (coin.v / coin.omega) * np.sin(coin.theta),
        coin.y + (coin.v / coin.omega) * np.cos(coin.theta)
    ], dtype=np.float64)
    d_i: fp64 = np.linalg.norm(_arc_center).astype(np.float64)
    if d_i == 0.0:
        raise InitialConditionError("Degenerate arc: center coincides with billiard center.")

    # Endpoint circle (radius r about C) must transversally cross the wall,
    # i.e. |R - r| <= d <= R + r.
    if d_i + r_i < b.R:
        raise InitialConditionError(
            "Endpoint circle is fully inscribed in the billiard; it never reaches "
            f"the wall (R={float(b.R)}, r={float(r_i)}, d={float(d_i)})."
        )
    if d_i + b.R < r_i or d_i > b.R + r_i:
        raise InitialConditionError(
            "Endpoint circle does not intersect the billiard wall "
            f"(R={float(b.R)}, r={float(r_i)}, d={float(d_i)})."
        )

    # 3. compute alpha_i, 4. beta_i (args in [-1, 1] given the checks above)
    alpha_arg = (d_i**2 + b.R**2 - r_i**2) / (2*d_i*b.R)
    beta_arg = (d_i**2 - b.R**2 + r_i**2) / (2*d_i*r_i)
    alpha_i: fp64 = np.arccos(np.clip(alpha_arg, -1.0, 1.0))
    beta_i: fp64 = np.arccos(np.clip(beta_arg, -1.0, 1.0))
    # 4.2 compute gamma_i and correct beta when the impact flipped the
    gamma_i: fp64 = np.arcsin(coin.l / r_i)     # Changed gamma defn to match 0615 file

    # 5 Helper: compute psi as center-point-angle for later use
    psi_i = normalize_angle(np.arctan2(_arc_center[1], _arc_center[0]))
    return (r_i, d_i, alpha_i, beta_i, gamma_i, psi_i)


def free_space_transition(
    coin_after_last_impact: Coin,
    billiard: Billiard,
    last_impact: ImpactData,
    interp_dt=1/30,
) -> Tuple[List[Coin], FreeSpaceArcData]:
    """
    Compute the free space transition between two coin configurations.
    - i.e. the arc connecting the start and end coin configurations.
    """
    coin: Coin = coin_after_last_impact
    b: Billiard = billiard

    # 0. Condition checks
    if not billiard.contains_coin(coin_after_last_impact):
        raise SimConsistencyError("Coin configuration after last impact must be inside billiard.")
    if coin.omega == 0.0:
        return _free_space_transition_line(coin, b, interp_dt)

    (r_i, d_i, alpha_i, beta_i, gamma_i, psi_i) = _free_space_params(coin, b)

    # 5. Swept rotation to the next impact (Prop. 6 / Lemma 4).
    _omgs = np.sign(coin.omega)
    if coin.v == 0.0:
        # v=0 (Sec 3.3): both endpoints trace the circle of radius l about the
        # fixed midpoint (r=l, gamma=pi/2); the opposite endpoint impacts next.
        swept: fp64 = 2.0 * beta_i - np.pi
        front_r = float(np.linalg.norm(coin.pos_front))
        back_r = float(np.linalg.norm(coin.pos_back))
        sigma_last = fp64(1.0) if abs(front_r - b.R) <= abs(back_r - b.R) else fp64(-1.0)
        contact_sign: fp64 = -sigma_last
    elif last_impact.is_leading_impact:
        # rho_l: trailing endpoint impacts next
        swept = 2.0 * (beta_i + gamma_i - np.pi)
        contact_sign = -np.sign(coin.v)
    else:
        # rho_T: leading endpoint impacts next
        swept = 2.0 * (beta_i - gamma_i)
        contact_sign = np.sign(coin.v)
    if swept <= 0.0:
        raise SimConsistencyError(
            f"Non-positive swept angle ({float(swept)}); degenerate impact geometry."
        )
    phi_next: fp64 = normalize_angle(psi_i - _omgs*alpha_i)
    # 6. Compute theta_{i+1}
    theta_next: fp64 = normalize_angle(coin.theta + _omgs*swept)
    # 7. Next impact time
    t_next_impact: fp64 = coin.t + swept / np.abs(coin.omega)

    # 8. Recover next coin cartesian configuration
    pos_next: ndarray = _impact_polar2cartesian(contact_sign, theta_next, phi_next, coin.l, b.R)

    # 9. Formulate next coin configuration
    next_coin: Coin = deepcopy(coin)
    next_coin.x = pos_next[0]
    next_coin.y = pos_next[1]
    next_coin.theta = theta_next
    next_coin.t = t_next_impact

    # Double check: recovered state must be a valid impact configuration
    if not billiard.contains_coin_boundary(next_coin):
        raise SimConsistencyError("Recovered next state is not a valid impact configuration.")

    # 10. Formulate free-space arc data for plotting
    free_space_arc_data = FreeSpaceArcData(
        r_i=r_i,
        d_i=d_i,
        alpha_i=alpha_i,
        beta_i=beta_i,
        psi=psi_i,
        phi_next=phi_next,
        theta_next=theta_next
    )

    # 11. Optional: interpolate free-space trajectory
    if interp_dt <= 0.0:
        raise InitialConditionError("Interpolation time step must be positive.")
    traj = _free_space_interpolate(
        coin_after_last_impact,
        next_coin,
        psi=psi_i,
        d_i=d_i,
        dt=interp_dt
    )
    # Assert all coins are in billiard
    for coin_i in traj:
        if not billiard.contains_coin(coin_i):
            raise SimConsistencyError(f"Coin configuration {coin_i} is outside the billiard.")
    return traj, free_space_arc_data


def _free_space_transition_line(
    coin: Coin,
    b: Billiard,
    interp_dt=1/30,
    tol=1e-9,
) -> Tuple[List[Coin], FreeSpaceArcData]:
    """
    Singular case omega = 0 (Sec 3.2): straight-line flight (Eq 3.21), map
    rho_v (Lemma 8). Delta t follows Prop 9 up to a sign correction; see the
    derivation: the leading point s + (2l + |v|t) e(theta) reaches the wall
    when 2l + |v|t = -2 sgn(v) R cos(phi - theta).
    """
    if coin.v == 0.0:
        raise InitialConditionError("(omega, v) = (0, 0) has trivial dynamics; excluded.")

    sgv = np.sign(coin.v)
    # A post-impact omega=0 state has its trailing endpoint on the wall (Sec 3.2.2).
    trailing = coin.pos - sgv * coin.l * \
        np.array([np.cos(coin.theta), np.sin(coin.theta)], dtype=np.float64)
    if abs(float(np.linalg.norm(trailing)) - float(b.R)) > 1e-7:
        raise SimConsistencyError(
            "omega=0 free flight requires the trailing endpoint on the wall."
        )
    phi_i = np.arctan2(trailing[1], trailing[0])
    cos_pt: fp64 = np.cos(phi_i - coin.theta)
    if coin.v * cos_pt >= -tol:
        raise SimConsistencyError(
            "omega=0 state is not post-impact: contact point not moving inward (Eq 3.23)."
        )

    # Prop 9 (corrected sign): time to the next impact, at the leading endpoint.
    t_next_impact: fp64 = coin.t + 2.0 * (-sgv * b.R * cos_pt - coin.l) / np.abs(coin.v)
    if t_next_impact <= coin.t:
        raise SimConsistencyError("Non-positive omega=0 flight time; degenerate geometry.")
    # Lemma 8: theta is constant, phi reflects as in the classical billiard.
    theta_next: fp64 = normalize_angle(coin.theta)
    phi_next: fp64 = normalize_angle(-phi_i + 2.0 * coin.theta - sgv * np.pi)

    pos_next: ndarray = _impact_polar2cartesian(sgv, theta_next, phi_next, coin.l, b.R)
    next_coin: Coin = deepcopy(coin)
    next_coin.x = pos_next[0]
    next_coin.y = pos_next[1]
    next_coin.theta = theta_next
    next_coin.t = t_next_impact
    if not b.contains_coin_boundary(next_coin):
        raise SimConsistencyError("Recovered next state is not a valid impact configuration.")

    free_space_arc_data = FreeSpaceArcData(
        r_i=fp64(np.inf),
        d_i=fp64(np.nan),
        alpha_i=fp64(np.nan),
        beta_i=fp64(np.nan),
        psi=fp64(np.nan),
        phi_next=phi_next,
        theta_next=theta_next,
    )

    if interp_dt <= 0.0:
        raise InitialConditionError("Interpolation time step must be positive.")
    traj = _line_interpolate(coin, next_coin, dt=interp_dt)
    for coin_i in traj:
        if not b.contains_coin(coin_i):
            raise SimConsistencyError(f"Coin configuration {coin_i} is outside the billiard.")
    return traj, free_space_arc_data


def first_impact(
    coin: Coin,
    billiard: Billiard,
    interp_dt=1/30
) -> Tuple[List[Coin], Optional[FreeSpaceArcData]]:
    """
    Given arbitrary starting configuration, check if it is valid, then interpolate to first impact point. 
    """
    # 0. Condition checks
    # If it is already an impact, just return
    if billiard.contains_coin_boundary(coin):
        return [coin], None
    # Otherwise, check if it is inside the billiard
    if not billiard.contains_coin(coin):
        raise InitialConditionError("Initial coin configuration must be inside billiard.")
    if coin.omega == 0.0:
        return _first_impact_line(coin, billiard, interp_dt)

    # Geometry params; also guards the no-first-impact cases (inscribed /
    # non-intersecting endpoint circle).
    (r_0, d_0, alpha_0, beta_0, gamma_0, psi_0) = _free_space_params(coin, billiard)

    # Proceed
    # 1. Forward sweep to the near-arc exit at intersection (psi+pi)+sgn(omega)*beta.
    theta_0 = coin.theta
    _omgs = np.sign(coin.omega)
    target = psi_0 + np.pi + _omgs*beta_0
    if coin.v == 0.0:
        # Pure rotation: front/back symmetric. The contact endpoint is fixed by
        # the rotation sense -- front hits first iff sgn(omega) sin(theta-psi) < 0
        # -- and its angular position about C=p is theta (front) or theta+pi (back).
        front_first = _omgs * np.sin(theta_0 - psi_0) < 0
        sigma = fp64(1.0) if front_first else fp64(-1.0)
        mu = theta_0 if front_first else theta_0 + np.pi
        swept = float((_omgs * (target - mu)) % (2*np.pi))
    else:
        # The leading endpoint (sign(v)) always hits first from the interior.
        endpoint_offset = np.pi if coin.v < 0 else 0.0
        lam0 = theta_0 + endpoint_offset - _omgs*np.pi/2 + _omgs*gamma_0
        swept = float((_omgs * (target - lam0)) % (2*np.pi))
        sigma = np.sign(coin.v)
    t_1 = swept / np.abs(coin.omega)

    # 2. Recover x, y, theta at first impact
    theta_1 = normalize_angle(theta_0 + _omgs*swept)
    x_1 = d_0*np.cos(psi_0) + (coin.v/coin.omega)*np.sin(theta_1)
    y_1 = d_0*np.sin(psi_0) - (coin.v/coin.omega)*np.cos(theta_1)

    # 3. phi_i: wall angle of the contact endpoint
    phi_1 = np.arctan2(
        y_1 + sigma*coin.l*np.sin(theta_1),
        x_1 + sigma*coin.l*np.cos(theta_1),
    )

    # 4. Formulate first-impact coin configuration
    first_impact_coin: Coin = deepcopy(coin)
    first_impact_coin.x = x_1
    first_impact_coin.y = y_1
    first_impact_coin.theta = theta_1
    first_impact_coin.t = t_1

    # 5. Formulate free-space arc data for plotting
    free_space_arc_data = FreeSpaceArcData(
        r_i=r_0,
        d_i=d_0,
        alpha_i=alpha_0,
        beta_i=beta_0,
        psi=psi_0,
        phi_next=phi_1,
        theta_next=theta_1
    )

    # 6. Interpolate from initial config to first impact
    if interp_dt <= 0.0:
        raise InitialConditionError("Interpolation time step must be positive.")
    traj = _free_space_interpolate(
        cs=coin,
        ce=first_impact_coin,
        psi=psi_0,
        d_i=d_0,
        dt=interp_dt
    )
    # Assert all coins are in billiard
    for coin_i in traj:
        if not billiard.contains_coin(coin_i):
            raise SimConsistencyError("Interpolated coin configuration must be inside billiard.")
    return traj, free_space_arc_data


def _first_impact_line(
    coin: Coin,
    b: Billiard,
    interp_dt=1/30,
) -> Tuple[List[Coin], Optional[FreeSpaceArcData]]:
    """
    Singular case omega = 0 from the interior (Sec 3.2): straight-line motion
    (Eq 3.21); the leading endpoint reaches the wall first. Its offset u along
    e(theta) from the midpoint start solves u^2 + 2u(p.e) + d^2 - R^2 = 0.
    """
    if coin.v == 0.0:
        raise InitialConditionError("(omega, v) = (0, 0) has trivial dynamics; excluded.")

    sgv = np.sign(coin.v)
    e_dir: ndarray = np.array([np.cos(coin.theta), np.sin(coin.theta)], dtype=np.float64)
    a: fp64 = coin.pos @ e_dir
    u_exit: fp64 = -a + sgv * np.sqrt(a**2 + b.R**2 - (coin.x**2 + coin.y**2))
    t_1: fp64 = (u_exit - sgv * coin.l) / coin.v
    if t_1 <= 0.0:
        raise SimConsistencyError("Non-positive omega=0 first-impact time.")

    pos_1: ndarray = coin.pos + coin.v * t_1 * e_dir
    exit_pt: ndarray = coin.pos + u_exit * e_dir
    phi_1: fp64 = np.arctan2(exit_pt[1], exit_pt[0])

    first_impact_coin: Coin = deepcopy(coin)
    first_impact_coin.x = pos_1[0]
    first_impact_coin.y = pos_1[1]
    first_impact_coin.t = coin.t + t_1
    if not b.contains_coin_boundary(first_impact_coin):
        raise SimConsistencyError("Recovered first-impact state is not a valid impact configuration.")

    free_space_arc_data = FreeSpaceArcData(
        r_i=fp64(np.inf),
        d_i=fp64(np.nan),
        alpha_i=fp64(np.nan),
        beta_i=fp64(np.nan),
        psi=fp64(np.nan),
        phi_next=phi_1,
        theta_next=normalize_angle(coin.theta),
    )

    if interp_dt <= 0.0:
        raise InitialConditionError("Interpolation time step must be positive.")
    traj = _line_interpolate(coin, first_impact_coin, dt=interp_dt)
    for coin_i in traj:
        if not b.contains_coin(coin_i):
            raise SimConsistencyError("Interpolated coin configuration must be inside billiard.")
    return traj, free_space_arc_data


def _line_interpolate(cs: Coin, ce: Coin, dt=1/30):
    """
    Interpolate between two coin configurations along a straight line (omega = 0).
    """
    if cs.t >= ce.t:
        raise SimConsistencyError("Start time must be less than end time.")
    if ce.t - cs.t <= dt:
        return [cs, ce]

    time_seq: ndarray = np.arange(cs.t, ce.t, dt) - cs.t
    coin_seq: List[Coin] = []
    for t in time_seq:
        coin_i = deepcopy(cs)
        coin_i.x = cs.x + cs.v * np.cos(cs.theta) * t
        coin_i.y = cs.y + cs.v * np.sin(cs.theta) * t
        coin_i.t = cs.t + t
        coin_seq.append(coin_i)
    coin_seq.append(ce)
    return coin_seq


def _free_space_interpolate(
    cs: Coin, 
    ce: Coin,
    psi: fp64,
    d_i: fp64,
    dt = 1/30
):
    """
    Interpolate between two coin configurations along an arc.
    """
    if cs.omega == 0.0:
        raise InitialConditionError("Cannot interpolate for zero angular velocity.")
    if cs.t >= ce.t:
        raise SimConsistencyError("Start time must be less than end time.")
    if ce.t - cs.t <= dt:
        return [cs, ce]

    time_seq: ndarray = np.arange(cs.t, ce.t, dt) - cs.t
    coin_seq: List[Coin] = []

    # Free-space-dynamics
    theta_seq: ndarray = normalize_angle(cs.theta + cs.omega * time_seq)
    x_seq: ndarray = d_i * np.cos(psi) + (cs.v/cs.omega) * np.sin(theta_seq)
    y_seq: ndarray = d_i * np.sin(psi) - (cs.v/cs.omega) * np.cos(theta_seq)

    for i, t in enumerate(time_seq):
        coin_i = deepcopy(cs)
        coin_i.x = x_seq[i]
        coin_i.y = y_seq[i]
        coin_i.theta = theta_seq[i]
        coin_i.t = cs.t + t
        coin_seq.append(coin_i)
    coin_seq.append(ce)
    return coin_seq
