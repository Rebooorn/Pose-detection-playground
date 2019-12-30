import numpy as np
import json
import matplotlib.pyplot as plt
from helper import *
import cv2

# This is not used, however is important for 3D plot
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as a3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#IMG_SIZE = [3384, 2710]
IMG_SIZE = [2710, 3384, 3]

# TEST_LOC_PARAM =  [0.214898, -0.0210617, -3.13819, -3.72663, 4.38063, 20.736]
TEST_LOC_PARAM =  [0.144469, -0.052166, -3.10855, 7.71139, 7.11818, 32.8131]

def car_model_render(vertex, faces):
    # path = os.path.join(os.path.dirname(__file__), 'data', 'car_models', '019-SUV.pkl')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    z = [np.ones(vertex.shape[0]), vertex[:,2]]
    z = np.array(z)
    ax.plot_surface(vertex[:, 0], vertex[:, 1], z)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_zlim(-2.5, 2.5)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for i in range(faces.shape[0]):
        vert = np.array([vertex[faces[i, 0] - 1],
                         vertex[faces[i, 1] - 1],
                         vertex[faces[i, 2] - 1]])
        ax.add_collection3d(Poly3DCollection([vert]))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_zlim(-2.5, 2.5)
    plt.show()


def car_model_2D_project(vertices, faces):

    camera_mtx = get_intrinsic_mtx()
    trans_mtx = get_translation_mtx(TEST_LOC_PARAM)
    pts_2d = get_2D_projection(pts_3d=vertices,
                               trans_mtx=camera_mtx @ trans_mtx)
    pts_2d = pts_2d.astype(np.int16)

    # display the points on the blank image, which should show the shape and loc of the car
    img = np.zeros(IMG_SIZE, np.uint8)
    # img = np.ones(IMG_SIZE, np.uint8) * 255
    for i in range(pts_2d.shape[0]):
        img[pts_2d[i,1], pts_2d[i,0]] = np.array([255, 255, 255])

    pts_2d = pts_2d[:, :2]
    # render the car with faces

    for face in faces:
        pt1 = (pts_2d[face[0] - 1][0], pts_2d[face[0] - 1][1])
        pt2 = (pts_2d[face[1] - 1][0], pts_2d[face[1] - 1][1])
        pt3 = (pts_2d[face[2] - 1][0], pts_2d[face[2] - 1][1])
        triangle_cnt = np.array([pt1, pt2, pt3], dtype=np.int64)
        cv2.drawContours(img, [triangle_cnt], 0, (255, 255, 255), -1)

    # load original img
    ori = cv2.imread(r'./data_sample/ID_00b7fb303.jpg')
    ori = cv2.cvtColor(ori, cv2.COLOR_BGR2RGB)

    # display the img
    cv2.imshow('car', cv2.resize(img, (800, 600)))
    cv2.imshow('ori', cv2.resize(ori, (800, 600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def contour_test():
    image = np.ones((300, 300, 3), np.uint8) * 255

    pt1 = (150, 100)
    pt2 = (100, 200)
    pt3 = (200, 200)

    triangle_cnt = np.array([pt1, pt2, pt3])

    cv2.drawContours(image, [triangle_cnt], 0, (0, 255, 0), -1)

    cv2.imshow("image", image)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    with open('linken-SUV.json', 'rb') as jf:
        car_model = json.load(jf)

    # load the vertex of car model
    vertex = car_model['vertices']
    vertex = np.array(vertex, dtype=np.float16)
    print(vertex.shape)

    faces = car_model['faces']
    faces = np.array(faces)
    print(faces.shape)

    # test to render 3D car model
    # car_model_render(vertex, faces)

    # test to project the 3D model to image
    # Tip: the z axis is defined in the opposite direction
    # vertex[:, 2] = -vertex[:, 2]

    car_model_2D_project(vertex, faces)

    # contour test
    # contour_test()
