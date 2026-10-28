"""Visual path targets embedded in their corresponding real observation."""
import numpy as np


def compose_goal_observation(frame, mask, observation):
    """Copy goal pixels inside the mask and real context outside it as uint8."""
    frame, mask, observation = np.asarray(frame), np.asarray(mask), np.asarray(observation)
    if frame.ndim != 2 or frame.shape != mask.shape or frame.shape != observation.shape:
        raise ValueError('Goal, mask and real observation must share 2D canvas geometry')
    result = np.array(observation, dtype=np.uint8, copy=True)
    selected = mask.astype(bool)
    result[selected] = frame[selected]
    return result
