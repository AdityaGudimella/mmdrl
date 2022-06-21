import numpy as np


def linearly_decaying_epsilon(
    decay_period: float, step: int, warmup_steps: int, epsilon: float
):
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0.0, 1 - epsilon)
    return epsilon + bonus


class DQNAgent:
    pass
