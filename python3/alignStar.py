#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Author: Jiajie Zhang
Email: zhangjiajie043@gmail.com
Update: Sean Liu
Email: sean.liu.2004@gmail.com
"""



import logging
import os
from math import log
import cv2
import numpy as np
import numpy.linalg as la
from scipy.spatial import distance as spd
import pywt
import Tyf
import tifffile as tiff
import DataModel

logging_level = logging.DEBUG
logging_format = "%(asctime)s (%(name)s) [%(levelname)s] line %(lineno)d: %(message)s"
logging.basicConfig(format=logging_format, level=logging_level)

data_model = DataModel.DataModel()
#img_tmpl = u"DSC028{:2d}.tif"
img_tmpl = u"DSC027{:2d}.ARW"

for p in [img_tmpl.format(i) for i in (21,20,22,23)]:
    logging.debug("image: %s", p)
    data_model.add_image(p)

ref_img = data_model.images[0]
#f = ref_img.focal_len
f=16.5
img_shape = ref_img.fullsize_gray_image.shape
img_size = np.array([img_shape[1], img_shape[0]])
data_model.reset_result()

pts, vol = DataModel.ImageProcessing.detect_star_points(ref_img.fullsize_gray_image)
sph = DataModel.ImageProcessing.convert_to_spherical_coord(pts, np.array((img_shape[1], img_shape[0])), f)
feature = DataModel.ImageProcessing.extract_point_features(sph, vol)
ref_img.features["pts"] = pts
ref_img.features["sph"] = sph
ref_img.features["vol"] = vol
ref_img.features["feature"] = feature

data_model.final_sky_img = np.copy(ref_img.original_image).astype("float32") / np.iinfo(
    ref_img.original_image.dtype).max

# Initial sum image
sky_imgs=[np.copy(data_model.final_sky_img)]
serial=0
result_img = (data_model.final_sky_img * np.iinfo("uint16").max).astype("uint16")
DataModel.ImageProcessing.save_tif_image("test00.tif", result_img, data_model.images[0].exif_info)

for img in data_model.images[1:]:
    pts, vol = DataModel.ImageProcessing.detect_star_points(img.fullsize_gray_image)
    sph = DataModel.ImageProcessing.convert_to_spherical_coord(pts, img_size, f)
    feature = DataModel.ImageProcessing.extract_point_features(sph, vol)
    img.features["pts"] = pts
    img.features["sph"] = sph
    img.features["vol"] = vol
    img.features["feature"] = feature

    pair_idx = DataModel.ImageProcessing.find_initial_match(img.features, ref_img.features)
    tf, pair_idx = DataModel.ImageProcessing.fine_tune_transform(img.features, ref_img.features, pair_idx)
    img_tf = cv2.warpPerspective(img.original_image, tf[0], tuple(img_size))
    img_tf = img_tf.astype("float32") / np.iinfo(img_tf.dtype).max
    serial=serial+1
    sky_imgs.append(np.copy(img_tf))
    result_img = (img_tf * np.iinfo("uint16").max).astype("uint16")
    DataModel.ImageProcessing.save_tif_image("test{:02d}.tif".format(serial), result_img, data_model.images[0].exif_info)


#data_model.final_sky_img = sum_img/(serial+1)
data_model.final_sky_img = np.mean(np.asarray(sky_imgs),axis=0)

result_img = (data_model.final_sky_img * np.iinfo("uint16").max).astype("uint16")
DataModel.ImageProcessing.save_tif_image("test.tif", result_img, data_model.images[0].exif_info)

