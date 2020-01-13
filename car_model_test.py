import numpy as np
import json, pickle, glob
import matplotlib.pyplot as plt
from helper import *
import cv2, os

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

# test of opencv drawContours() function
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

# TEST: reading csv file using python
def csv_reading_test(csv_path):
    train_params = dict()
    with open(csv_path) as f:
        reader = csv.reader(f, delimiter=',')
        # skip the first row
        reader.__next__()
        for row in reader:
            # split the string by space
            train_param_tmp = np.reshape(row[1].split(), (-1, 7))
            train_params[row[0]] = train_param_tmp

    # pickle dump the dict
    with open('train_pose_params.pickle', 'wb') as f:
        pickle.dump(train_params, f)

def car_ID_model_list():
    # match the ID to the model of the car, and save the dict to pickle
    model_list = [
             'baojun-310-2017',
                'biaozhi-3008',
          'biaozhi-liangxiang',
           'bieke-yinglang-XT',
                'biyadi-2x-F0',
               'changanbenben',
                'dongfeng-DS5',
                     'feiyate',
         'fengtian-liangxiang',
                'fengtian-MPV',
           'jilixiongmao-2015',
           'lingmu-aotuo-2009',
                'lingmu-swift',
             'lingmu-SX4-2012',
              'sikeda-jingrui',
        'fengtian-weichi-2006',
                   '037-CAR02',
                     'aodi-a6',
                   'baoma-330',
                   'baoma-530',
            'baoshijie-paoche',
             'bentian-fengfan',
                 'biaozhi-408',
                 'biaozhi-508',
                'bieke-kaiyue',
                        'fute',
                     'haima-3',
               'kaidilake-CTS',
                   'leikesasi',
               'mazida-6-2015',
                  'MG-GT-2015',
                       'oubao',
                        'qiya',
                 'rongwei-750',
                  'supai-2016',
             'xiandai-suonata',
            'yiqi-benteng-b50',
                       'bieke',
                   'biyadi-F3',
                  'biyadi-qin',
                     'dazhong',
              'dazhongmaiteng',
                    'dihao-EV',
      'dongfeng-xuetielong-C6',
     'dongnan-V3-lingyue-2011',
    'dongfeng-yulong-naruijie',
                     '019-SUV',
                   '036-CAR01',
                 'aodi-Q7-SUV',
                  'baojun-510',
                    'baoma-X5',
             'baoshijie-kayan',
             'beiqi-huansu-H3',
              'benchi-GLK-300',
                'benchi-ML500',
         'fengtian-puladuo-06',
            'fengtian-SUV-gai',
    'guangqi-chuanqi-GS4-2015',
        'jianghuai-ruifeng-S3',
                  'jili-boyue',
                      'jipu-3',
                  'linken-SUV',
                   'lufeng-X8',
                 'qirui-ruihu',
                 'rongwei-RX5',
             'sanling-oulande',
                  'sikeda-SUV',
            'Skoda_Fabia-2011',
            'xiandai-i25-2016',
            'yingfeinidi-qx80',
             'yingfeinidi-SUV',
                  'benchi-SUR',
                 'biyadi-tang',
           'changan-CS35-2012',
                 'changan-cs5',
          'changcheng-H6-2016',
                 'dazhong-SUV',
     'dongfeng-fengguang-S560',
       'dongfeng-fengxing-SX6',
    ]
    print(model_list[-10])

    with open('car_ID_model.pickle', 'wb') as f:
        pickle.dump(model_list, f)


def training_dataset_generate(TrainImgPath, TrainMaskPath, CarModelPath):
    # load original image and pose parameters, and generate the mask for training
    with open('train_pose_params.pickle', 'rb') as f:
        car_param_dict = pickle.load(f)

    with open('car_ID_model.pickle', 'rb') as f:
        car_ID_list = pickle.load(f)

    train_imgs = glob.glob(TrainImgPath + '/*.jpg')
    for img_path in train_imgs:
        ID = os.path.basename(img_path)[:-4]
        car_list = car_param_dict[ID]

        # create the original mask, which is completely black
        mask = np.zeros(IMG_SIZE, np.uint8)

        # each car_param means one car
        for car_param in car_list:
            # get car ID and .json car model
            car_ID =car_ID_list[np.int(car_param[0])]
            with open(CarModelPath + '/' + car_ID +'.json', 'rb') as jf:
                car_model = json.load(jf)

            # get pose info
            car_pose = [ np.float(n) for n in car_param[1:]]

            # Draw the 2D car onto the mask
            mask = generate_training_mask(img = mask,
                                          car_model = car_model,
                                          pose = car_pose)

        # After drawing all cars, save the image
        ori = cv2.resize(cv2.imread(img_path), (800, 800))
        ori = cv2.addWeighted(ori, 0.7, cv2.resize(mask, (800, 800)), 0.3, 0)
        cv2.imshow('ori', ori)
        cv2.imshow('mask', cv2.resize(mask, (800, 800)))
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    with open('019-SUV.json', 'rb') as jf:
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

    # car_model_2D_project(vertex, faces)

    # contour test
    # contour_test()

    # csv reading test
    # csv_path = r'D:\\liuchang\\Kaggle_proj\\12212019\\Pose-detection-playground\\data_sample\\train.csv'
    # csv_reading_test(csv_path)

    # test to match car ID and car model
    # car_ID_model_list()

    training_dataset_generate(TrainImgPath=r"D:\liuchang\Kaggle_proj\12212019\data\train_images",
                              TrainMaskPath=r"D:\liuchang\Kaggle_proj\12212019\data\train_grand_truths",
                              CarModelPath=r'D:\liuchang\Kaggle_proj\12212019\data\car_models_json')