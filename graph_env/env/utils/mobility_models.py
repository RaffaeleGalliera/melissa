from abc import ABC, abstractmethod
import numpy as np
from . import constants


class MobilityModel(ABC):
    """Base class for mobility models.

    Each model maintains its own RNG and current positions,
    and must implement `step(dt)` to advance by `dt` time units
    and return an (N,2) array of new positions.
    """
    def __init__(self, rng: np.random.RandomState, pos0: np.ndarray):
        self.rng = rng
        # ensure float32, shape (N,2)
        self.pos = pos0.astype(np.float32)

    @abstractmethod
    def step(self, dt: float) -> np.ndarray:
        """Advance the model by dt and return updated positions."""
        ...


class RandomWaypoint(MobilityModel):
    """Memoryless “straight‐to‐waypoint” mobility with optional pauses.

    Speeds are normalized so that `v_min`/`v_max` represent movement per step
    regardless of dt, by dividing by the node‐movement step constant.
    """

    def __init__(
        self,
        rng: np.random.RandomState,
        pos0: np.ndarray,
        v_min: float = 0.01,
        v_max: float = 0.05,
        pause_mean: float = 0.0,
        area: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    ):
        super().__init__(rng, pos0)
        self.v_min = v_min
        self.v_max = v_max
        self.pause_mean = pause_mean
        self.xmin, self.xmax, self.ymin, self.ymax = area
        n = pos0.shape[0]

        # normalize speeds so travel per call equals v_min..v_max
        scale = constants.NODES_MOVEMENT_STEP or 1.0
        raw = rng.uniform(self.v_min, self.v_max, size=n)
        self.speeds = (raw / scale).astype(np.float32)
        self.remaining_pause = np.zeros(n, dtype=np.float32)
        self.waypoints = self._sample_waypoints(n)

    def _sample_waypoints(self, n: int) -> np.ndarray:
        xs = self.rng.uniform(self.xmin, self.xmax, size=n)
        ys = self.rng.uniform(self.ymin, self.ymax, size=n)
        return np.stack([xs, ys], axis=1).astype(np.float32)

    def step(self, dt: float) -> np.ndarray:
        delta = self.waypoints - self.pos
        dist = np.linalg.norm(delta, axis=1)
        dir_unit = np.zeros_like(delta)
        nonzero = dist > 0
        dir_unit[nonzero] = (delta[nonzero] / dist[nonzero, None])

        # pauses
        moving = self.remaining_pause <= 0
        self.remaining_pause[~moving] -= dt
        dir_unit[~moving] = 0.0
        speeds = self.speeds * moving.astype(np.float32)

        # travel distance = speed * dt / scale => raw speed
        travel = np.minimum(speeds * dt, dist)
        self.pos += (dir_unit * travel[:, None])

        # arrivals
        arrived = (dist <= 1e-6) & moving
        if np.any(arrived):
            idx = np.nonzero(arrived)[0]
            self.waypoints[idx] = self._sample_waypoints(len(idx))
            raw = self.rng.uniform(self.v_min, self.v_max, len(idx))
            self.speeds[idx] = (raw / (constants.NODES_MOVEMENT_STEP or 1.0)).astype(np.float32)
            if self.pause_mean > 0:
                self.remaining_pause[idx] = self.rng.exponential(self.pause_mean, size=len(idx))

        # clip
        self.pos[:, 0] = np.clip(self.pos[:, 0], self.xmin, self.xmax)
        self.pos[:, 1] = np.clip(self.pos[:, 1], self.ymin, self.ymax)
        return self.pos


class GaussMarkov(MobilityModel):
    """Each call to .step() moves ~mean_speed units toward a correlated
       direction, then reflects at the borders."""

    def __init__(
        self,
        rng: np.random.RandomState,
        pos0: np.ndarray,
        alpha: float = 0.85,
        mean_speed: float = None,
        sigma: float = 0.20,
        area: tuple[float,float,float,float] = (0,1,0,1),
    ):
        super().__init__(rng, pos0)
        self.alpha      = alpha
        # default to using one node‐movement‐step as the average speed
        self.mean_speed = mean_speed if mean_speed is not None else constants.NODES_MOVEMENT_STEP
        self.sigma = sigma * self.mean_speed
        self.xmin,self.xmax,self.ymin,self.ymax = area

        n = pos0.shape[0]
        # initialize random directions with magnitude = mean_speed
        theta = rng.uniform(0, 2*np.pi, size=n)
        self.v = np.stack([np.cos(theta), np.sin(theta)], axis=1) \
                 * self.mean_speed

    def _reflect(self):
        x,y = self.pos[:,0], self.pos[:,1]
        left, right   = x < self.xmin, x > self.xmax
        bottom, top   = y < self.ymin, y > self.ymax
        # bounce
        self.v[left|right, 0] *= -1
        self.v[bottom|top, 1] *= -1
        # clamp
        self.pos[:,0] = np.clip(self.pos[:,0], self.xmin, self.xmax)
        self.pos[:,1] = np.clip(self.pos[:,1], self.ymin, self.ymax)

    def step(self, dt: float) -> np.ndarray:
        # -- update velocity (AR(1) + noise) --
        noise = self.rng.normal(0, self.sigma, size=self.v.shape)
        norms = np.linalg.norm(noise, axis=1, keepdims=True)
        unit = np.where(norms>0, noise/norms, 0.0)

        self.v = (
            self.alpha * self.v
            + (1 - self.alpha) * self.mean_speed * unit
            + np.sqrt(1 - self.alpha**2) * noise
        )

        self.pos += self.v

        self._reflect()
        return self.pos