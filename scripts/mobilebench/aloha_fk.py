"""Forward kinematics of the RMBench / RoboTwin aloha-agilex arms (numpy, no simulator).

Chain constants are copied from RMBench_assets/embodiments/aloha-agilex/urdf/
arx5_description_isaac.urdf (fl_* = left arm, fr_* = right arm). Poses are expressed in
the robot's `footprint` frame, which is the body frame B_t of the MobileBench design
(fixed for this robot). The TCP is

    R_tcp = R_link6,   p_tcp = p_link6 + R_link6 @ [0.12, 0, 0]

which reproduces the recorded `endpose/{left,right}_endpose` of the raw RMBench episodes
once mapped to the world by robot_pose (p + [0, -0.65, 0], 90 deg about z): 0.8 mm mean /
4 mm max position error and 0.1 deg rotation error over raw episodes 0-9 (the residual
is joint_action vs. the measured joints).
"""

import numpy as np

GRIPPER_BIAS = 0.12

# (xyz, rpy, axis) per joint, base_joint first (fixed), then joint1..joint6 (revolute).
_ARM = {
    "left": [
        ((0.2305, 0.297, 0.782), (0.0, 0.0, 0.02), None),
    ],
    "right": [
        ((0.2315, -0.3063, 0.781), (0.0, 0.0, 0.01), None),
    ],
}
_SHARED = [
    ((0, 0, 0.058), (0, 0, 0), (0, 0, 1)),
    ((0.025013, 0.00060169, 0.042), (0, 0, 0), (0, 1, 0)),
    ((-0.26396, 0.0044548, 0), (-3.1416, 0, -0.015928), (0, 1, 0)),
    ((0.246, -0.00025, -0.06), (0, 0, 0), (0, 1, 0)),
    ((0.06775, 0.0015, -0.0855), (0, 0, -0.015928), (0, 0, 1)),
    ((0.03095, 0, 0.0855), (-3.1416, 0, 0), (1, 0, 0)),
]
for _k in _ARM:
    _ARM[_k] = _ARM[_k] + _SHARED

def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_angle(axis: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Batched rotation about a fixed unit axis, q [N] -> [N, 3, 3]."""
    x, y, z = axis
    c, s = np.cos(q), np.sin(q)
    t = 1 - c
    return np.stack(
        [
            np.stack([t * x * x + c, t * x * y - s * z, t * x * z + s * y], -1),
            np.stack([t * x * y + s * z, t * y * y + c, t * y * z - s * x], -1),
            np.stack([t * x * z - s * y, t * y * z + s * x, t * z * z + c], -1),
        ],
        -2,
    )


def tcp_pose(arm: str, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """q [N, 6] joint angles -> (pos [N, 3], rot [N, 3, 3]) of the TCP in the footprint frame."""
    q = np.asarray(q, dtype=np.float64).reshape(-1, 6)
    n = q.shape[0]
    rot = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    pos = np.zeros((n, 3))
    for i, (xyz, rpy, axis) in enumerate(_ARM[arm]):
        pos = pos + rot @ np.asarray(xyz, dtype=np.float64)
        rot = rot @ _rpy(*rpy)
        if axis is not None:
            rot = rot @ _axis_angle(np.asarray(axis, dtype=np.float64), q[:, i - 1])
    pos = pos + rot[:, :, 0] * GRIPPER_BIAS
    return pos, rot


def both_tcp(state14: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[N, 14] aloha state (left6, lgrip, right6, rgrip) -> pos [N, 2, 3], rot [N, 2, 3, 3] (0=left, 1=right)."""
    s = np.asarray(state14).reshape(-1, 14)
    pl, rl = tcp_pose("left", s[:, 0:6])
    pr, rr = tcp_pose("right", s[:, 7:13])
    return np.stack([pl, pr], 1), np.stack([rl, rr], 1)
