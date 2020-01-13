import numpy as np
import json, pickle, glob
import matplotlib.pyplot as plt
from helper import *
import cv2, os

IMG_SIZE = [2710, 3384, 3]

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
    training_dataset_generate(TrainImgPath=r'.\\data_sample\\train_image',
                              TrainMaskPath=r".\\data_sample\\train_masks",
                              CarModelPath=r'.\\data_sample\\car_models_json')