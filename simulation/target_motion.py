"""
目标运动模型: 静态、匀速直线、随机转向
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DT, WORLD_SIZE


class TargetMotionModel:

    def __init__(self, init_pos, init_vel=None, mode='static',
                 dt=None, process_noise_std=0.1, turn_rate_std=None,
                 speed=1.0, seed=42):
        self.dt = dt or DT
        self.mode = mode
        self.speed = speed
        self.process_noise_std = process_noise_std
        self.turn_rate_std = turn_rate_std or np.deg2rad(15)
        self._rng = np.random.RandomState(seed)

        self.state = np.zeros(4)
        self.state[:2] = np.asarray(init_pos, dtype=float)
        if init_vel is not None:
            self.state[2:] = np.asarray(init_vel, dtype=float)
        elif mode == 'linear':
            heading = self._rng.uniform(0, 2 * np.pi)
            self.state[2] = speed * np.cos(heading)
            self.state[3] = speed * np.sin(heading)
        elif mode == 'random_turn':
            heading = self._rng.uniform(0, 2 * np.pi)
            self.state[2] = speed * np.cos(heading)
            self.state[3] = speed * np.sin(heading)

        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        q = self.process_noise_std ** 2
        dt2 = self.dt ** 2
        dt3 = self.dt ** 3
        dt4 = self.dt ** 4
        self.Q = q * np.array([
            [dt4 / 4, 0, dt3 / 2, 0],
            [0, dt4 / 4, 0, dt3 / 2],
            [dt3 / 2, 0, dt2, 0],
            [0, dt3 / 2, 0, dt2],
        ], dtype=float)

    def step(self):
        if self.mode == 'static':
            return self.state[:2].copy()

        if self.mode == 'random_turn':
            heading = np.arctan2(self.state[3], self.state[2])
            heading += self._rng.normal(0, self.turn_rate_std)
            self.state[2] = self.speed * np.cos(heading)
            self.state[3] = self.speed * np.sin(heading)

        self.state = self.F @ self.state
        noise = self._rng.multivariate_normal(np.zeros(4), self.Q)
        self.state += noise

        margin = 2.0
        for dim in range(2):
            if self.state[dim] < margin:
                self.state[dim] = margin
                self.state[dim + 2] = abs(self.state[dim + 2])
            elif self.state[dim] > WORLD_SIZE - margin:
                self.state[dim] = WORLD_SIZE - margin
                self.state[dim + 2] = -abs(self.state[dim + 2])

        return self.state[:2].copy()

    def get_pos(self):
        return self.state[:2].copy()

    def get_state(self):
        return self.state.copy()

    def get_F(self):
        return self.F.copy()

    def get_Q(self):
        return self.Q.copy()
