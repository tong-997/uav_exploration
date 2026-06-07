"""
目标跟踪有限状态机: 单机视角的任务状态管理
"""
import numpy as np

STATES = [
    'SEARCH', 'CONFIRM_TARGET', 'TRACK_TARGET', 'OCCLUDED_TRACK',
    'REACQUIRE_TARGET', 'CONVERGE_TO_TARGET', 'MULTI_TRACK',
    'ENCIRCLE_TARGET', 'LOST_TARGET', 'FINISH',
]


class TargetTrackingFSM:

    def __init__(self, uav_id, confirmation_frames=3,
                 reacquire_timeout=50):
        self.uav_id = uav_id
        self.confirmation_frames = confirmation_frames
        self.reacquire_timeout = reacquire_timeout

        self.state = 'SEARCH'
        self.consecutive_detections = 0
        self.target_confirmed = False
        self._reacquire_start = -1
        self.state_history = []

    def transition(self, step, detected, kf_track_state,
                   n_tracking_uavs=0, at_slot=False, all_at_slots=False):
        prev = self.state

        if self.state == 'SEARCH':
            if detected:
                self.state = 'CONFIRM_TARGET'
                self.consecutive_detections = 1

        elif self.state == 'CONFIRM_TARGET':
            if detected:
                self.consecutive_detections += 1
                if self.consecutive_detections >= self.confirmation_frames:
                    self.state = 'TRACK_TARGET'
                    self.target_confirmed = True
            else:
                self.consecutive_detections = 0
                self.state = 'SEARCH'

        elif self.state == 'TRACK_TARGET':
            if kf_track_state == 'LOST':
                self.state = 'REACQUIRE_TARGET'
                self._reacquire_start = step
            elif kf_track_state == 'OCCLUDED_TRACK':
                self.state = 'OCCLUDED_TRACK'
            elif n_tracking_uavs >= 2:
                self.state = 'MULTI_TRACK'

        elif self.state == 'OCCLUDED_TRACK':
            if detected:
                self.state = 'TRACK_TARGET'
            elif kf_track_state == 'LOST':
                self.state = 'REACQUIRE_TARGET'
                self._reacquire_start = step

        elif self.state == 'REACQUIRE_TARGET':
            if detected:
                self.state = 'TRACK_TARGET'
                self._reacquire_start = -1
            elif self._reacquire_start > 0 and \
                    step - self._reacquire_start > self.reacquire_timeout:
                self.state = 'LOST_TARGET'

        elif self.state == 'CONVERGE_TO_TARGET':
            if detected and self.target_confirmed:
                if n_tracking_uavs >= 2:
                    self.state = 'MULTI_TRACK'
                else:
                    self.state = 'TRACK_TARGET'
            elif kf_track_state == 'LOST':
                self.state = 'LOST_TARGET'

        elif self.state == 'MULTI_TRACK':
            if kf_track_state == 'LOST':
                self.state = 'REACQUIRE_TARGET'
                self._reacquire_start = step
            elif all_at_slots:
                self.state = 'ENCIRCLE_TARGET'

        elif self.state == 'ENCIRCLE_TARGET':
            if kf_track_state == 'LOST':
                self.state = 'REACQUIRE_TARGET'
                self._reacquire_start = step
            elif all_at_slots:
                self.state = 'FINISH'

        elif self.state == 'LOST_TARGET':
            if detected:
                self.state = 'CONFIRM_TARGET'
                self.consecutive_detections = 1

        if self.state != prev:
            self.state_history.append((step, prev, self.state))

        return self.state

    def receive_broadcast(self, step):
        if self.state == 'SEARCH':
            self.state = 'CONVERGE_TO_TARGET'
            self.state_history.append((step, 'SEARCH', 'CONVERGE_TO_TARGET'))

    def get_action(self, target_pos_est, slot_pos, uav_pos):
        if self.state in ('SEARCH', 'FINISH'):
            return None
        if self.state in ('CONFIRM_TARGET', 'TRACK_TARGET',
                          'OCCLUDED_TRACK', 'REACQUIRE_TARGET'):
            return target_pos_est
        if self.state in ('CONVERGE_TO_TARGET', 'MULTI_TRACK',
                          'ENCIRCLE_TARGET'):
            return slot_pos if slot_pos is not None else target_pos_est
        return None
