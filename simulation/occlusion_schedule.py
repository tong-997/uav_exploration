"""
遮挡时间表: 控制目标在不同仿真时段的遮挡等级
"""

DEFAULT_SCHEDULE = [
    (120, 160, 'light'),
    (220, 270, 'medium'),
    (330, 380, 'heavy'),
    (430, 470, 'heavy'),
]

OCCLUSION_VALUES = {
    'none': 0.0,
    'light': 0.2,
    'medium': 0.5,
    'heavy': 0.85,
    'full': 1.0,
}


class OcclusionSchedule:

    def __init__(self, mode='no_occlusion', schedule=None):
        self.mode = mode
        self.schedule = schedule or DEFAULT_SCHEDULE

    def get_level(self, step):
        if self.mode == 'no_occlusion':
            return 'none'

        if self.mode == 'partial_occlusion':
            return 'medium'

        if self.mode == 'intermittent_occlusion':
            for start, end, level in self.schedule:
                if start <= step <= end:
                    return level
            return 'none'

        if self.mode == 'full_occlusion_interval':
            for start, end, level in self.schedule:
                if start <= step <= end:
                    if level == 'heavy':
                        return 'full'
                    return level
            return 'none'

        return 'none'

    def get_occlusion_value(self, step):
        return OCCLUSION_VALUES.get(self.get_level(step), 0.0)
