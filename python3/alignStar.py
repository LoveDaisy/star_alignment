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
import argparse

def reorder_images(images):
    ''' reorder_images
        Input: a list of image names
        Return: a list, first item is the one in the middle,the rest in sorted order
    '''
    sortedImages=sorted(images)
    ref=sortedImages[len(images)//2]
    newlist=[ref]
    sortedImages.remove(ref)
    return newlist+sortedImages

parser = argparse.ArgumentParser(description='Align stars in the sky')
parser.add_argument('images', nargs='+',
                    help='A list of image files')
parser.add_argument('-o', '--output', default="aligned.tif" ,
                    help='Output file')
parser.add_argument('-d', '--debug', action='store_true',default=False,
                    help='Output file')
parser.add_argument('-k','--keepInterim', action='store_true',
                    default=False,
                    help='Keep all interim images')
parser.add_argument('-f', '--focalLength', dest='focal', type=float,
                    help='Focal length (default: 11)')
parser.add_argument('-c', '--cropFactor', dest='cropFactor', type=float, default=1.0,
                    help='Crop factor (default: 1.0)')

args = parser.parse_args()

if args.debug:
    logging_level = logging.DEBUG
    logging_format = "%(asctime)s (%(name)s) [%(levelname)s] line %(lineno)d: %(message)s"
    logging.basicConfig(format=logging_format, level=logging_level)

keepInterim=args.keepInterim
data_model = DataModel.DataModel()
outputName=args.output

for p in reorder_images(args.images):
    logging.debug("image: %s", p)
    data_model.add_image(p)

ref_img = data_model.images[0]
f=args.focal
crop_factor=args.cropFactor
logging.debug("Setting focal length to %f",f)

img_shape = ref_img.fullsize_gray_image.shape
img_size = np.array([img_shape[1], img_shape[0]])
data_model.reset_result()

pts, vol = DataModel.ImageProcessing.detect_star_points(ref_img.fullsize_gray_image)
sph = DataModel.ImageProcessing.convert_to_spherical_coord(pts, np.array((img_shape[1], img_shape[0])), f,crop_factor)
feature = DataModel.ImageProcessing.extract_point_features(sph, vol)
ref_img.features["pts"] = pts
ref_img.features["sph"] = sph
ref_img.features["vol"] = vol
ref_img.features["feature"] = feature

data_model.final_sky_img = np.copy(ref_img.original_image).astype("float32") / np.iinfo(
    ref_img.original_image.dtype).max

# Initialize aligned image list
#sky_imgs=[np.copy(data_model.final_sky_img)]
sum_img=np.copy(data_model.final_sky_img)
serial=0
if keepInterim:
    result_img = (data_model.final_sky_img * np.iinfo("uint16").max).astype("uint16")
    DataModel.ImageProcessing.save_tif_image("interim00.tif", result_img, data_model.images[0].exif_info)

for img in data_model.images[1:]:
    pts, vol = DataModel.ImageProcessing.detect_star_points(img.fullsize_gray_image)
    sph = DataModel.ImageProcessing.convert_to_spherical_coord(pts, img_size, f, crop_factor)
    feature = DataModel.ImageProcessing.extract_point_features(sph, vol)
    img.features["pts"] = pts
    img.features["sph"] = sph
    img.features["vol"] = vol
    img.features["feature"] = feature

    try:
        pair_idx = DataModel.ImageProcessing.find_initial_match(img.features, ref_img.features)
        tf, pair_idx = DataModel.ImageProcessing.fine_tune_transform(img.features, ref_img.features, pair_idx)
        img_tf = cv2.warpPerspective(img.original_image, tf[0], tuple(img_size))
        img_tf = img_tf.astype("float32") / np.iinfo(img_tf.dtype).max
        sum_img=sum_img+img_tf
        serial+=1
        if keepInterim:
            result_img = (img_tf * np.iinfo("uint16").max).astype("uint16")
            DataModel.ImageProcessing.save_tif_image("interim{:02d}.tif".format(serial), result_img, data_model.images[0].exif_info)
    except ValueError as e:
        print("Alignment failed for this picture:",str(e),". Discarded")

data_model.final_sky_img = sum_img/(serial+1)
#data_model.final_sky_img = np.mean(np.asarray(sky_imgs),axis=0)

result_img = (data_model.final_sky_img * np.iinfo("uint16").max).astype("uint16")
DataModel.ImageProcessing.save_tif_image(outputName, result_img, data_model.images[0].exif_info)
print ("Done. Output image in ",outputName)
