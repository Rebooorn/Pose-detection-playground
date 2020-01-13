import numpy as np
import csv, cv2


def get_intrinsic_mtx():
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
    yaw, pitch, roll = np.pi-pitch, np.pi-yaw, -roll

    mtx = np.array([
        [np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll), x],
        [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll), y],
        [-np.sin(pitch),            np.cos(pitch)*np.sin(roll),                                      np.cos(pitch)*np.cos(roll),                                      z],
        [0,                         0,                                                               0,                                                               1]
    ])

    return mtx

def get_2D_projection(pts_3d, trans_mtx):
    # Get the 2D projection of 3D point clouds pts_3d, using the transform matrix trans_mtx.
    # pts_3d.shape = [n, 3]
    # trans_mtx.shape = [3, 4]

    # return shape: pts_2d.T.shape = [n, 2]

    # convert 3d to affine coordinate
    pts_3d_affine = np.concatenate((pts_3d, np.ones([pts_3d.shape[0], 1])), axis=1)
    pts_2d_affine = trans_mtx @ pts_3d_affine.T

    # convert 2d affine to 2d image coordinate
    tmp = np.stack((pts_2d_affine[-1, :], pts_2d_affine[-1, :], pts_2d_affine[-1, :]))
    # tmp = np.array([[pts_2d_affine[-1, :]],[pts_2d_affine[-1, :]],[pts_2d_affine[-1, :]]])
    pts_2d = pts_2d_affine / tmp

    return pts_2d.T



def generate_training_mask(img, car_model, pose, disp=False):
    # Helper to generate the mask for the training
    # img: the image where the object area will be assigned to 255. Img should have 3 channels
    # car_model: json model of one car
    # pose: 6D pose information

    vertices = car_model['vertices']
    faces = car_model['faces']

    vertices = np.array(vertices, dtype=np.float16)
    faces = np.array(faces)

    camera_mtx = get_intrinsic_mtx()
    trans_mtx = get_translation_mtx(pose)

    pts_2d = get_2D_projection(pts_3d=vertices,
                               trans_mtx=camera_mtx @ trans_mtx)
    pts_2d = pts_2d.astype(np.int16)

    for face in faces:
        pt1 = (pts_2d[face[0] - 1][0], pts_2d[face[0] - 1][1])
        pt2 = (pts_2d[face[1] - 1][0], pts_2d[face[1] - 1][1])
        pt3 = (pts_2d[face[2] - 1][0], pts_2d[face[2] - 1][1])
        triangle_cnt = np.array([pt1, pt2, pt3], dtype=np.int64)
        cv2.drawContours(img, [triangle_cnt], 0, (0, 0, 255), -1)

    if disp:
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img







