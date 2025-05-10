import enum

import numpy as np


class Label(enum.IntEnum):
    FAILURE = 0  # grasp execution failed due to collision or slippage
    SUCCESS = 1  # object was successfully removed


class Grasp(object):
    """Grasp parameterized as pose of a 2-finger robot hand.
    
    NOTE(mbreyer): clarify definition of grasp frame
    """

    def __init__(self, pose, width):
        self.pose = pose
        self.width = width


def to_voxel_coordinates(grasp, voxel_size):
    pose = grasp.pose
    pose.translation /= voxel_size
    width = grasp.width / voxel_size
    return Grasp(pose, width)


def from_voxel_coordinates(grasp, voxel_size):
    pose = grasp.pose
    pose.translation *= voxel_size
    # pose.translation = 0.4 - pose.translation
    width = grasp.width * voxel_size
    return Grasp(pose, width)


if __name__ == '__main__':
    from src.gd.utils.transform import Transform, Rotation
    grasps = []
    for i in range(10):
        grasps.append(Grasp(Transform(Rotation.identity(), [0., 0., 0.]), 0.5))
    print(type(grasps))
    print(type(np.asarray(grasps)))

