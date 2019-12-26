import numpy as np


def get_intrinsic_mtx(path):
    # camera intrinsic parameters are defined by the existing camera
    fx = 2304.5479
    fy = 2305.8757
    cx = 1686.2379
    cy = 1354.9849

    mtx = np.array([[fx, 0, cx, 0],
                    [0, fy, cy, 0],
                    [0,  0,  1, 0]])

    return mtx

def get_translation_mtx(param):
    # The param is the list: [pitch, yaw, roll, x, y, z]
    # return the translation mtx from the parameters
    pitch, yaw, roll, x, y, z = param
    mtx = np.array([
        [np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll), x],
        [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll), y],
        [-np.sin(pitch),            np.cos(pitch)*np.sin(roll),                                      np.cos(pitch)*np.cos(roll),                                      z],
        [0,                         0,                                                               0,                                                               1]
    ])
    return mtx

