# -*- coding: utf-8 -*-

from os import path
from sys import modules

import cv2
import numpy as np
from PIL import Image
from pkg_resources import resource_filename
from skimage.measure import label, regionprops
from skimage.transform import rotate, resize
from tensorflow.python.keras.preprocessing import image

import mnet.Model_DiscSeg
import mnet.Model_MNet
import mnet.mnet_utils

DiscROI_size = 600
Disc_size = 640
CDRSeg_size = 400

parent_dir = path.dirname(resource_filename(modules[__name__].__name__, '__init__.py'))

test_data_path = path.join(parent_dir, 'glaucoma-cases')
data_save_path = mnet.mnet_utils.mk_dir(path.join(parent_dir,
                                                  'glaucoma-cases'))

DiscSeg_model = mnet.Model_DiscSeg.DeepModel(size_set=Disc_size)
DiscSeg_model.load_weights(path.join(parent_dir,
                                     'mnet/deep_model',
                                     'Model_DiscSeg_ORIGA.h5'))

CDRSeg_model = mnet.Model_MNet.DeepModel(size_set=CDRSeg_size)
CDRSeg_model.load_weights(path.join(parent_dir,
                                    'mnet/deep_model',
                                    'Model_MNet_REFUGE.h5'))


def MNetMask(temp_txt):
    try:
        # load image
        org_img = np.asarray(image.load_img(path.join(test_data_path,
                                                      temp_txt)))
        # Disc region detection by U-Net
        temp_img = resize(org_img, (Disc_size, Disc_size, 3)) * 255
        temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
        disc_map = DiscSeg_model.predict([temp_img])
        disc_map = mnet.mnet_utils.BW_img(np.reshape(disc_map,
                                                     (Disc_size, Disc_size)),
                                          0.5)

        regions = regionprops(label(disc_map))
        C_x = int(regions[0].centroid[0] * org_img.shape[0] / Disc_size)
        C_y = int(regions[0].centroid[1] * org_img.shape[1] / Disc_size)
        disc_region, err_xy, crop_xy = mnet.mnet_utils.disc_crop(org_img, DiscROI_size, C_x, C_y)

        # Disc and Cup segmentation by M-Net
        Disc_flat = rotate(cv2.linearPolar(disc_region, (DiscROI_size / 2, DiscROI_size / 2),
                                           DiscROI_size / 2, cv2.WARP_FILL_OUTLIERS), -90)
        temp_img = mnet.mnet_utils.pro_process(Disc_flat, CDRSeg_size)
        temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
        [_, _, _, _, prob_10] = CDRSeg_model.predict(temp_img)

        # Extract mask
        prob_map = np.reshape(prob_10, (prob_10.shape[1], prob_10.shape[2], prob_10.shape[3]))
        disc_map = np.array(Image.fromarray(prob_map[:, :, 0]).resize((DiscROI_size, DiscROI_size)))
        cup_map = np.array(Image.fromarray(prob_map[:, :, 1]).resize((DiscROI_size, DiscROI_size)))
        disc_map[-round(DiscROI_size / 3):, :] = 0
        cup_map[-round(DiscROI_size / 2):, :] = 0
        De_disc_map = cv2.linearPolar(rotate(disc_map, 90),
                                      (DiscROI_size / 2, DiscROI_size / 2),
                                      DiscROI_size / 2,
                                      cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
        De_cup_map = cv2.linearPolar(rotate(cup_map, 90),
                                     (DiscROI_size / 2, DiscROI_size / 2),
                                     DiscROI_size / 2,
                                     cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)

        De_disc_map = np.array(mnet.mnet_utils.BW_img(De_disc_map, 0.5),
                               dtype=int)
        De_cup_map = np.array(mnet.mnet_utils.BW_img(De_cup_map, 0.5),
                              dtype=int)

        # Save raw mask
        ROI_result = np.array(mnet.mnet_utils.BW_img(De_disc_map, 0.5), dtype=int) + np.array(mnet.mnet_utils.BW_img(De_cup_map, 0.5), dtype=int)
        Img_result = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.int8)
        Img_result[crop_xy[0]:crop_xy[1], crop_xy[2]:crop_xy[3], ] = ROI_result[err_xy[0]:err_xy[1], err_xy[2]:err_xy[3], ]
        save_result = Image.fromarray((255 - Img_result * 127).astype(np.uint8))
        output = path.join(data_save_path, 'masks', temp_txt[:-4] + '.png')
        save_result.save(output)
        return output
    except:
        return False


if __name__ == '__main__':
    MNetMask('V0001.jpg')
