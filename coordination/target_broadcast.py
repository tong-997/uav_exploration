"""
目标状态广播: 模拟 UAV 间的目标信息通信
"""
import numpy as np
from dataclasses import dataclass, field
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import COMM_RANGE, DT


@dataclass
class TargetStatePacket:
    sender_id: int
    step: int
    target_pos_est: np.ndarray
    target_vel_est: np.ndarray
    cov: np.ndarray
    confidence: float
    track_state: str


class TargetBroadcast:

    def __init__(self, comm_range=None, delay_steps=1,
                 packet_loss_rate=0.0, seed=42):
        self.comm_range = comm_range or COMM_RANGE
        self.delay_steps = delay_steps
        self.packet_loss_rate = packet_loss_rate
        self._rng = np.random.RandomState(seed + 5000)
        self._buffer = {}

    def broadcast(self, sender_id, sender_pos, step,
                  kf_state_dict, all_uav_positions):
        packet = TargetStatePacket(
            sender_id=sender_id,
            step=step,
            target_pos_est=kf_state_dict['x_est'].copy(),
            target_vel_est=kf_state_dict['v_est'].copy(),
            cov=kf_state_dict['P'].copy(),
            confidence=kf_state_dict['confidence'],
            track_state=kf_state_dict['track_state'],
        )

        for j in range(len(all_uav_positions)):
            if j == sender_id:
                continue
            dist = np.linalg.norm(
                np.asarray(sender_pos) - np.asarray(all_uav_positions[j]))
            if dist > self.comm_range:
                continue
            if self._rng.random() < self.packet_loss_rate:
                continue
            if j not in self._buffer:
                self._buffer[j] = []
            self._buffer[j].append(packet)

    def receive(self, receiver_id, current_step):
        if receiver_id not in self._buffer:
            return []
        ready = [p for p in self._buffer[receiver_id]
                 if current_step - p.step >= self.delay_steps]
        self._buffer[receiver_id] = [
            p for p in self._buffer[receiver_id]
            if current_step - p.step < self.delay_steps]
        return ready

    @staticmethod
    def propagate_to_current(packet, current_step, dt=None):
        dt = dt or DT
        delta = current_step - packet.step
        if delta <= 0:
            return packet.target_pos_est.copy(), packet.cov[:2, :2].copy()
        F = np.array([[1, 0, dt, 0],
                       [0, 1, 0, dt],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], dtype=float)
        state = np.zeros(4)
        state[:2] = packet.target_pos_est
        state[2:] = packet.target_vel_est
        P = packet.cov.copy()
        Q_small = np.eye(4) * 0.01
        for _ in range(delta):
            state = F @ state
            P = F @ P @ F.T + Q_small
        return state[:2].copy(), P[:2, :2].copy()
