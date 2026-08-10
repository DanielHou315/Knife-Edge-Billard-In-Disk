from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy import float64 as fp64

@dataclass
class Billiard():
	R: fp64 = fp64(1.0)

	def contains_coin(self, coin: "Coin", tol=1e-9) -> bool:
		"""Returns True if the coin is inside (<=R) the billiard, False otherwise."""
		# Given circle, check that both front and back of coin are inside circle
		# If they are both in, then the coin is necessarily in billiard since convex.
		# Otherwise, there is an issue
		for p in [coin.pose_front, coin.pose_back]:
			if np.linalg.norm(p[:2]) > self.R + tol:
				return False
		return True

	def contains_coin_boundary(self, coin: "Coin", tol=1e-9) -> bool:
		"""Returns True if the coin is inside the billiard and an endpoint touches the wall, False otherwise."""
		if not self.contains_coin(coin, tol):
			return False
		front_dist = float(np.linalg.norm(coin.pos_front))
		back_dist = float(np.linalg.norm(coin.pos_back))

		# Either front or back within tol, but not both, so XOR
		touches = (abs(front_dist - self.R) <= tol) ^ (abs(back_dist - self.R) <= tol)
		return bool(touches)

@dataclass
class Coin:
	# q: pose
	x: fp64 = fp64(0.0)
	y: fp64 = fp64(0.0)
	theta: fp64 = fp64(0.0)

	# v: velocity
	v: fp64 = fp64(0.0)
	omega: fp64 = fp64(0.0)

	# Physical properties
	l: fp64 = fp64(1.0)		# radius of coin
	m: fp64 = fp64(1.0)		# mass of coin
	J: fp64 = fp64(1.0)		# moment of inertia of coin about its center of mass

	# time
	t: fp64 = fp64(0.0)		# time of current configuration

	@property
	def pos(self) -> np.ndarray:
		"""Returns the position of the coin."""
		return np.array([self.x, self.y], dtype=np.float64)

	@property
	def pos_front(self) -> np.ndarray:
		"""Returns the position of the front of the coin."""
		x_front: fp64 = self.x + self.l * np.cos(self.theta)
		y_front: fp64 = self.y + self.l * np.sin(self.theta)
		return np.array([x_front, y_front], dtype=np.float64)

	@property
	def pos_back(self) -> np.ndarray:
		"""Returns the position of the back of the coin."""
		x_back: fp64 = self.x - self.l * np.cos(self.theta)
		y_back: fp64 = self.y - self.l * np.sin(self.theta)
		return np.array([x_back, y_back], dtype=np.float64)

	@property
	def vel(self) -> np.ndarray:
		"""Returns the velocity of the coin."""
		return np.array([self.v, self.omega], dtype=np.float64)
	
	@property
	def pose(self) -> np.ndarray:
		"""Returns the pose of the coin."""
		return np.array([self.x, self.y, self.theta], dtype=np.float64)

	@property
	def pose_front(self) -> np.ndarray:
		"""Returns the pose of the front of the coin."""
		return np.array(
			[self.pos_front[0], self.pos_front[1], self.theta], 
		dtype=np.float64)

	@property
	def pose_back(self) -> np.ndarray:
		"""Returns the pose of the back of the coin."""
		return np.array(
			[self.pos_back[0], self.pos_back[1], self.theta],
		dtype=np.float64)
		
@dataclass
class ImpactData:
	impact_pt: np.ndarray	
	t: fp64  				# time of impact
	phi: fp64
	is_leading_impact: bool = False

@dataclass
class FreeSpaceArcData:
	r_i: fp64
	d_i: fp64
	alpha_i: fp64
	beta_i: fp64
	psi: fp64
	phi_next: fp64
	theta_next: fp64

	@property
	def arc_center(self) -> np.ndarray:
		"""Returns the center of the free-space arc."""
		x_center: fp64 = self.d_i * np.cos(self.psi)
		y_center: fp64 = self.d_i * np.sin(self.psi)
		return np.array([x_center, y_center], dtype=np.float64)

class ArcGeometry(NamedTuple):
    """Geometry of the contact-endpoint arc relative to the billiard wall."""
    r: fp64          # radius of the endpoint's free-space circle
    d: fp64          # distance from billiard centre to the arc centre C
    center: np.ndarray  # arc centre C
    alpha: fp64      # triangle angle at the origin O
    beta: fp64       # triangle angle at the arc centre C
    gamma: fp64      # angular offset midpoint -> contact endpoint, seen fromC


