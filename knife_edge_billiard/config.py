from pathlib import Path
from typing import Optional, Tuple

import yaml
from numpy import float64 as fp64
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import Billiard, Coin, normalize_angle, parse_angle


class BilliardConfig(BaseModel):
    """Validated billiard parameters that produce a :class:`Billiard` dataclass.

    Unknown keys are rejected and ``R`` must be strictly positive.  Call
    :meth:`to_billiard` to obtain the lightweight ``fp64`` dataclass used by the
    physics code.
    """

    model_config = ConfigDict(extra="forbid")

    R: float = Field(gt=0, description="Billiard radius (must be positive).")

    def to_billiard(self) -> Billiard:
        """Build the plain :class:`Billiard` dataclass with ``fp64`` fields."""
        return Billiard(R=fp64(self.R))


# Moment of inertia of a symmetric body of half-length ``l``, written as
# J = kappa * m * l^2.  The mass distribution is only constrained by the
# symmetry requirement (centre of mass at the midpoint), so kappa is not
# unique; it is however bounded above by the limiting case of all the mass
# sitting on the boundary, i.e. two point masses m/2 at the two endpoints:
#
#     kappa = 1   <=>   J = m * l^2      (all mass at the endpoints)
#
# Configurations violating that bound describe no rigid body and produce
# nonsensical trajectories.  The default is the uniform bar spanning
# [-l, +l], kappa = 1/3, equivalently J = m * ell^2 / 12 for the full length
# ell = 2 * l.
UNIFORM_BAR_KAPPA = 1.0 / 3.0
MAX_KAPPA = 1.0


class CoinConfig(BaseModel):
    """Validated coin parameters that produce a :class:`Coin` dataclass.

    Unknown keys are rejected and the physical properties (``l``, ``m``, ``J``)
    must be strictly positive.  ``theta`` may be given as a number or as an
    expression string in terms of ``pi`` (e.g. ``"pi/2"``); it is stored
    normalized to ``[-pi, pi]``.  The configuration time ``t`` is not
    user-configurable -- :meth:`to_coin` always starts the coin at ``t = 0``.
    Call :meth:`to_coin` to obtain the lightweight ``fp64`` dataclass used by
    the physics code.

    ``J`` is *not* an independent parameter: it is fixed by the mass and the
    geometry.  Omit it (the normal case) to get the uniform-bar value
    ``m * l**2 / 3``; supply it only to model a different symmetric mass
    distribution, in which case it must respect ``J <= m * l**2``.
    """

    model_config = ConfigDict(extra="forbid")

    # Pose
    x: float
    y: float
    theta: float

    # Velocity
    v: float
    omega: float

    # Physical properties (strictly positive)
    l: float = Field(gt=0, description="Coin half-length / radius.")
    m: float = Field(gt=0, description="Coin mass.")
    J: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Coin moment of inertia about its centre.  Omit to derive the "
            "uniform-bar value J = m*l^2/3; if given, must satisfy J <= m*l^2."
        ),
    )

    @field_validator("theta", mode="before")
    @classmethod
    def _parse_and_normalize_theta(cls, value):
        """Accept a number or a ``pi`` expression and normalize to [-pi, pi]."""
        return float(normalize_angle(parse_angle(value)))

    @model_validator(mode="after")
    def _inertia_within_physical_bound(self) -> "CoinConfig":
        """Reject an explicit ``J`` that no symmetric mass distribution realizes."""
        bound = MAX_KAPPA * self.m * self.l**2
        if self.J is not None and self.J > bound:
            raise ValueError(
                f"J={self.J} exceeds the physical bound m*l^2={bound} "
                f"(m={self.m}, l={self.l}); the maximum is attained only in the "
                "limit of all the mass sitting at the two endpoints."
            )
        return self

    @property
    def inertia(self) -> float:
        """The moment of inertia actually used: explicit ``J``, else uniform bar."""
        if self.J is not None:
            return self.J
        return UNIFORM_BAR_KAPPA * self.m * self.l**2

    def to_coin(self) -> Coin:
        """Build the plain :class:`Coin` dataclass with ``fp64`` fields.

        ``t`` is fixed at ``0.0``: a run always starts from the configured pose.
        ``J`` is taken from :attr:`inertia`, i.e. derived when not configured.
        """
        return Coin(
            x=fp64(self.x), y=fp64(self.y), theta=fp64(self.theta),
            v=fp64(self.v), omega=fp64(self.omega),
            l=fp64(self.l), m=fp64(self.m), J=fp64(self.inertia),
            t=fp64(0.0),
        )


class SimConfig(BaseModel):
    """A billiard together with a coin placed inside it.

    This is the composition root for config validation.  Cross-object rules
    (those needing both the billiard and the coin) live here as
    ``@model_validator``s.  For now there is exactly one such rule:

    1. Both coin endpoints lie inside the billiard.

    Wall-touching is intentionally *not* required: arbitrary interior
    placements are valid.  Whether a config already sits at an impact is a
    separate concern, answered by ``Billiard.contains_coin_boundary``.
    """

    model_config = ConfigDict(extra="forbid")

    billiard: BilliardConfig
    coin: CoinConfig

    @model_validator(mode="after")
    def _coin_inside_billiard(self) -> "SimConfig":
        billiard = self.billiard.to_billiard()
        coin = self.coin.to_coin()
        if not billiard.contains_coin(coin):
            raise ValueError(
                "coin is not fully inside the billiard "
                f"(R={float(billiard.R)}, front={coin.pos_front.tolist()}, "
                f"back={coin.pos_back.tolist()})."
            )
        return self

    @classmethod
    def from_yaml(cls, path) -> "SimConfig":
        """Load and validate a Billiard+Coin config from a YAML file.

        Raises ``FileNotFoundError`` for a missing file and ``ValueError`` for
        malformed YAML or any validation failure (pydantic's ``ValidationError``
        is a subclass of ``ValueError``).
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        try:
            data = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            raise ValueError(f"Malformed YAML in config file {path}: {exc}") from exc
        return cls.model_validate(data)

    def to_dataclasses(self) -> Tuple[Billiard, Coin]:
        """Return the validated ``(Billiard, Coin)`` dataclasses."""
        return self.billiard.to_billiard(), self.coin.to_coin()

    def pretty(self) -> str:
        """Return a human-readable YAML rendering of the validated config.

        ``J`` is rendered as the value the simulation will use, so a derived
        inertia shows up as a number rather than as ``null``.
        """
        data = self.model_dump()
        data["coin"]["J"] = self.coin.inertia
        return yaml.safe_dump(
            data, sort_keys=False, default_flow_style=False
        ).rstrip()
