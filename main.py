################################
# Title :- Gaussion HeatMap    #
# By    :- Paritosh Yadav      #
# Date  :- 21 Jan 2020         #
################################

import matplotlib.pyplot as plt
import numpy as np
import cv2
from shapely.geometry import Polygon
from os import listdir

def four_point_transform(image, pts):
    max_x, max_y = np.max(pts[:, 0]).astype(
        np.int32), np.max(pts[:, 1]).astype(np.int32)

    dst = np.array([
        [0, 0],
        [image.shape[1] - 1, 0],
        [image.shape[1] - 1, image.shape[0] - 1],
        [0, image.shape[0] - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(dst, pts)
    warped = cv2.warpPerspective(image, M, (max_x, max_y))

    return warped


class DataLoader():

    def __init__(self, BasePath):

        self.mat = f"{BasePath}/mat/"

        self.img = f"{BasePath}/img/"
        sigma = 10
        spread = 3
        extent = int(spread * sigma)
        self.gaussian_heatmap = np.zeros([2 * extent, 2 * extent], dtype=np.float32)

        for i in range(2 * extent):
            for j in range(2 * extent):
                self.gaussian_heatmap[i, j] = 1 / 2 / np.pi / (sigma ** 2) * np.exp(
                    -1 / 2 * ((i - spread * sigma - 0.5) ** 2 + (j - spread * sigma - 0.5) ** 2) / (sigma ** 2))
        self.gaussian_heatmap = (self.gaussian_heatmap / np.max(self.gaussian_heatmap) * 255).astype(np.uint8)


    def add_character(self, image, bbox):
        top_left = np.array([np.min(bbox[:, 0]), np.min(bbox[:, 1])]).astype(np.int32)
        if top_left[1] > image.shape[0] or top_left[0] > image.shape[1]:
            # This means there is some bug in the character bbox
            # Will have to look into more depth to understand this
            return image
        bbox -= top_left[None, :]
        transformed = four_point_transform(self.gaussian_heatmap.copy(), bbox.astype(np.float32))

        start_row = max(top_left[1], 0) - top_left[1]
        start_col = max(top_left[0], 0) - top_left[0]
        end_row = min(top_left[1] + transformed.shape[0], image.shape[0])
        end_col = min(top_left[0] + transformed.shape[1], image.shape[1])

        image[max(top_left[1], 0):end_row, max(top_left[0], 0):end_col] += transformed[start_row:end_row - top_left[1],
                                                                           start_col:end_col - top_left[0]]

        return image


    def generate_target(self, image_size, character_bbox):
        character_bbox = character_bbox.transpose(2, 1, 0)

        channel, height, width = image_size

        target = np.zeros([height, width], dtype=np.uint8)

        for i in range(character_bbox.shape[0]):
            target = self.add_character(target, character_bbox[i])

        return target / 255, np.float32(target != 0)


    def add_affinity(self, image, bbox_1, bbox_2):
        center_1, center_2 = np.mean(bbox_1, axis=0), np.mean(bbox_2, axis=0)
        tl = np.mean([bbox_1[0], bbox_1[1], center_1], axis=0)
        bl = np.mean([bbox_1[2], bbox_1[3], center_1], axis=0)
        tr = np.mean([bbox_2[0], bbox_2[1], center_2], axis=0)
        br = np.mean([bbox_2[2], bbox_2[3], center_2], axis=0)

        affinity = np.array([tl, tr, br, bl])

        return self.add_character(image, affinity)


    def generate_affinity(self, image_size, character_bbox, text):
        """

        :param image_size: shape = [3, image_height, image_width]
        :param character_bbox: [2, 4, num_characters]
        :param text: [num_words]
        :return:
        """

        character_bbox = character_bbox.transpose(2, 1, 0)

        channel, height, width = image_size

        target = np.zeros([height, width], dtype=np.uint8)

        total_letters = 0

        for word in text:
            for char_num in range(len(word) - 1):
                target = self.add_affinity(target, character_bbox[total_letters].copy(),
                                           character_bbox[total_letters + 1].copy())
                total_letters += 1
            total_letters += 1

        return target / 255, np.float32(target != 0)


    def __getitem__(self, filename):

        
        from scipy.io import loadmat

        mat = loadmat(self.mat + filename)
        imname, charBB, txt = mat["imnames"], mat["charBB"],mat["txt"]

        image = plt.imread(self.img + imname[0]).transpose(2, 0, 1) / 255
        
        weight, target = self.generate_target(image.shape, charBB.copy())
        weight_affinity, target_affinity = self.generate_affinity(image.shape, charBB.copy(), txt.copy())

        return image, weight, target, weight_affinity, target_affinity


if __name__ == "__main__":


# make sure you have this dir structure 
    BASEPATH="Data" # this folder contain MAT(mat),IMG(img) folder
    MATFILEPATH="Data/mat/"
    OUTPUTPATH="Data/output/"

    dataloader = DataLoader(BASEPATH)

    files = [f for f in listdir(MATFILEPATH) if f.split(".")[-1].lower() in ['mat','MAT']]
    for file in files:
        print(f"{file} is Processing")
        image, weight, target, weight_affinity, target_affinity = dataloader[file]
        file=file.split('.')[0][:-1]
        # plt.imsave(f'{OUTPUTPATH}/{file}_image.png', image.transpose(1, 2, 0))
        plt.imsave(f'{OUTPUTPATH}/{file}_target.png', target, cmap='gray')
        plt.imsave(f'{OUTPUTPATH}/{file}_weight.png', weight, cmap='gray')
        plt.imsave(f'{OUTPUTPATH}/{file}_weight_affinity.png', weight_affinity)
        plt.imsave(f'{OUTPUTPATH}/{file}_target_affinity.png', target_affinity)
        plt.imsave(f'{OUTPUTPATH}/{file}_together.png',np.concatenate([weight[:, :, None],
                                            weight_affinity[:, :, None],
                                            np.zeros_like(weight)[:, :, None]],
                                            axis=2))




