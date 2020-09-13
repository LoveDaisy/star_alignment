#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Author: Jiajie Zhang
Email: zhangjiajie043@gmail.com
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
import rawpy


class Image(object):
    """ Image class
        Attributes:
            full path
            original image, may be uint16 type
            fullsize gray image
            exif info, Tyf.TiffFile type
            image features
    """

    def __init__(self, full_path):
        super(Image, self).__init__()

        self.full_path = full_path
        self.dir, self.name = os.path.split(full_path)
        self.focal_len = None
        self.features = {}
        self.tf = None

        _, ext = os.path.splitext(full_path)
        if ext.lower() in (".tiff", ".tif") and os.path.isfile(full_path):
            self.original_image, self.exif_info = ImageProcessing.read_tif_image(full_path)
            gray_img = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
            self.fullsize_gray_image = ImageProcessing.convert_to_float(gray_img)
        elif ext.lower() in (".nef", ".arw", "cr2") and os.path.isfile(full_path):
            self.original_image, self.exif_info = ImageProcessing.read_raw_image(full_path)
            gray_img = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
            self.fullsize_gray_image = ImageProcessing.convert_to_float(gray_img)
        else:
            self.original_image = None
            self.fullsize_gray_image = None
            self.exif_info = None

        self.reset_all()

    def reset_focal_length(self):
        f = self.get_exif_value("FocalLength")
        if f and len(f) == 2:
            self.focal_len = f[0] * 1.0 / f[1]
        elif f and len(f) == 1:
            self.focal_len = f[0]
        else:
            self.focal_len = None

    def reset_all(self):
        self.reset_focal_length()
        self.features = {}
        self.tf = None

    def get_exif_value(self, name):
        if not self.exif_info:
            return None

        info = self.exif_info[0].find(name)
        if not info:
            return None
        else:
            return info.value


class DataModel(object):
    # Align options
    AUTO_MASK = 1
    ALIGN_STARS = 2
    ALIGN_GROUND = 3

    # Display options
    ORIGINAL_IMAGE = 1

    def __init__(self):
        super(DataModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.images = []
        self.ref_ind = 0
        self.image_dir = None

        self.final_sky_img = None       # Of type double
        self.final_ground_img = None    # Of type double

        # For concurrency issue
        self.is_adding_image = False

        # Other GUI options
        self.merge_option_type = self.ALIGN_STARS

    def add_image(self, path):
        self.logger.debug("add_image()")
        img_dir, name = os.path.split(path)

        if not os.path.exists(path) or not os.path.isfile(path):
            self.logger.error("File %s does not exist!", path)
            return False

        for img in self.images:
            if path == img.full_path:
                self.logger.info("Image already exists. File: %s", path)
                return False

        if self.is_adding_image:
            return False

        self.is_adding_image = True
        img = Image(path)
        focal_len = img.get_exif_value("FocalLength")
        self.images.append(img)
        self.logger.debug("Loading image %s... Focal length = %s", name, focal_len)
        if not self.image_dir:
            self.image_dir = img_dir
        self.is_adding_image = False
        return True

    def update_final_sky(self, img):
        self.logger.debug("update_final_sky()")
        self.final_sky_num += 1
        if self.final_sky_img is None and self.final_sky_num == 1:
            self.final_sky_img = np.copy(img)
        elif self.final_sky_img is not None and self.final_sky_num > 0:
            # self.final_sky_img = np.fmax(self.final_sky_img, img)
            self.final_sky_img = self.final_sky_img / self.final_sky_num * (self.final_sky_num - 1) + img / self.final_sky_num

    def update_final_ground(self, img):
        self.logger.debug("update_final_ground()")
        self.final_ground_num += 1
        if self.final_ground_img is None and self.final_ground_num == 1:
            self.final_ground_img = np.copy(img)
        elif self.final_ground_img is not None and self.final_ground_num > 0:
            self.final_ground_img = self.final_ground_img / self.final_ground_num * (self.final_ground_num - 1) + img / self.final_ground_num

    def clear_images(self):
        self.logger.debug("clear_images()")
        self.images = []
        self.reset_final_sky()
        self.reset_final_ground()
        self.image_dir = None
        self.ref_ind = 0
        self.is_adding_image = False

    def reset_final_sky(self):
        self.logger.debug("reset_final_sky()")
        self.final_sky_img = None
        self.final_sky_num = 0

    def reset_final_ground(self):
        self.logger.debug("reset_final_ground()")
        self.final_ground_img = None
        self.final_ground_num = 0

    def reset_result(self):
        self.logger.debug("reset_result()")
        self.reset_final_sky()
        self.reset_final_ground()
        for img in self.images:
            img.features = {}

    def has_image(self):
        res = len(self.images) > 0
        self.logger.debug("has_image(): %s", res)
        return res

    def iter_images(self):
        self.logger.debug("iter_images()")
        return iter(self.images)

    def total_images(self):
        res = len(self.images)
        self.logger.debug("total_images(): %s", res)
        return res

    def has_sky_result(self):
        res = self.final_sky_img is not None
        self.logger.debug("has_sky_result(): %s", res)
        return res

    def has_ground_result(self):
        res = self.final_ground_img is not None
        self.logger.debug("has_ground_result(): %s", res)
        return res


class ImageProcessing(object):
    def __init__(self):
        super(ImageProcessing, self).__init__()

    @staticmethod
    def wavelet_dec_rec(img_blr, resize_factor=0.25):
        img_shape = img_blr.shape

        need_resize = abs(resize_factor - 1) > 0.001
        level = int(6 - log(1 / resize_factor, 2))

        if need_resize:
            img_blr_resize = cv2.resize(img_blr, None, fx=resize_factor, fy=resize_factor)
        else:
            img_blr_resize = img_blr
        coeffs = pywt.wavedec2(img_blr_resize, "db8", level=level)

        coeffs[0].fill(0)
        coeffs[-1][0].fill(0)
        coeffs[-1][1].fill(0)
        coeffs[-1][2].fill(0)

        img_rec_resize = pywt.waverec2(coeffs, "db8")
        if need_resize:
            img_rec = cv2.resize(img_rec_resize, (img_shape[1], img_shape[0]))
        else:
            img_rec = img_rec_resize

        return img_rec

    @staticmethod
    def detect_star_points(img_gray, mask=None, resize_length=2200):
        logging.debug("detect_star_point()")
        logging.debug("resize_length = %s", resize_length)

        sigma = 2
        img_shape = img_gray.shape
        img_blr = cv2.GaussianBlur(img_gray, (9, 9), sigma)
        img_blr_mean=np.mean(img_blr)
        img_blr_range=np.max(img_blr) - np.min(img_blr)
        img_blr = (img_blr - img_blr_mean) / img_blr_range
 
        resize_factor = 1
        while max(img_shape) * resize_factor > resize_length:
            resize_factor /= 2

        logging.debug("Calculating mask...")

        tmp_mask = cv2.resize(img_gray, None, fx=resize_factor, fy=resize_factor)
        tmp_mask_10percent=np.percentile(tmp_mask, 10)
        tmp_mask = (tmp_mask < min(tmp_mask_10percent, 0.15)).astype(np.uint8) * 255
##        tmp_mask = np.logical_and(tmp_mask < np.percentile(tmp_mask, 10), tmp_mask < 0.15).astype(np.uint8) * 255
        logging.debug("Mask logical selection complete")
        dilate_size = int(max(img_shape) * 0.02 * resize_factor * 5)
        tmp_mask = 255 - cv2.dilate(tmp_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size)))
        tmp_mask = cv2.resize(tmp_mask, (img_shape[1], img_shape[0]))
        if mask is None:
            mask = tmp_mask > 127
        else:
            mask = np.logical_and(tmp_mask > 127, mask > 0)
        logging.debug("Mask calculation Complete")

        mask_rate = np.sum(mask) * 100.0 / np.prod(mask.shape)
        logging.debug("mask rate: %.2f", mask_rate)
        if mask_rate < 50:
            mask = np.ones(tmp_mask.shape, dtype="bool")
        
        while True:
            try:
##                img_rec = ImageProcessing.wavelet_dec_rec(contrast_img, resize_factor=resize_factor) * mask
                img_rec = ImageProcessing.wavelet_dec_rec(img_blr, resize_factor=resize_factor) * mask

                bw = ((img_rec > np.percentile(img_rec[mask], 99.5)) * mask).astype(np.uint8) * 255
                bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

                contours, _ = cv2.findContours(np.copy(bw), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                contours = [contour for contour in contours if len(contour) > 5]
                logging.debug("%d star points detected", len(contours))

                if len(contours) > 400:
                    break
                else:
                    raise ValueError("No enough points")
            except ValueError as e:
                if resize_factor >= 1:
                    raise ValueError("Cannot detect enough star points:" + str(e))
                else:
                    resize_factor *= 2
        logging.debug("resize factor = %f", resize_factor)

        #elps - match contours to an ellipse. Return a Box2D - coordinates of a rotated rectangle - 3x tuples
        #first tuple is the center of the box, the second gives the width and the height and the last is the angle.
        elps = [cv2.fitEllipse(contour) for contour in contours]
        #centroids - the "center" of the ellipses
        centroids = np.array([e[0] for e in elps])
        #areas - the areas of the contours, but 0.5*len(contour)?
        areas = np.array([cv2.contourArea(contour) + 0.5 * len(contour) for contour in contours])
        #eccentricities - how irregular the ellipse is. 
        eccentricities = np.sqrt(np.array([1 - (elp[1][0] / elp[1][1]) ** 2 for elp in elps]))

        # Calculate "intensity"
        
        mask = np.zeros(bw.shape, np.uint8)
        intensities = np.zeros(areas.shape)
        for i in range(len(contours)):
            cv2.drawContours(mask, contours[i], 0, 255, -1)
            #It is a straight rectangle, it doesn't consider the rotation of the object. .
            #Let (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
            #x,y,w,h = cv2.boundingRect(cnt)
            rect = cv2.boundingRect(contours[i])
            val = cv2.mean(img_rec[rect[1]:rect[1] + rect[3] + 1, rect[0]:rect[0] + rect[2] + 1],
                           mask[rect[1]:rect[1] + rect[3] + 1, rect[0]:rect[0] + rect[2] + 1])
            mask[rect[1]:rect[1] + rect[3] + 1, rect[0]:rect[0] + rect[2] + 1] = 0
            intensities[i] = val[0]

        valid_stars = np.logical_and(areas > 20, areas < 200, eccentricities < .8)
        valid_stars = np.logical_and(valid_stars, areas > np.percentile(areas, 20), intensities > np.percentile(intensities, 20))

        star_pts = centroids[valid_stars]  # [x, y]
        print("Final star points = {0}".format(len(star_pts)))

        areas = areas[valid_stars]
        intensities = intensities[valid_stars]

        return star_pts, areas * intensities

    @staticmethod
    def convert_to_spherical_coord(star_pts, img_size, focal_length,crop_factor=1.0):
        '''
        convert_to_spherical_coord
        Input:
        start_pts: np array of start points in x,y coodinates
        img_size: image size in pixels
        focal_length: focal length, the "real focal length" before crop factor. In real life no real effect observed.
        crop_factor: sensor crop factor. In real life no real effect is observed.
        Output: theta and phi in spherical corrdinates.
        '''
        logging.debug("convert_coord_img_sph(Focal length {0})".format(focal_length))
        FullFrameSensorWidth=36                     #Full frame sensor width is 36mm
        sensorSize=FullFrameSensorWidth/crop_factor #Actual sensor size
        DPI = np.max(img_size)/sensorSize
        
        p0 = (star_pts - img_size / 2.0)            #Adjust start x,y coords to the middle of lens
        p=p0/DPI

##        p0 = (star_pts - img_size / 2.0) / (np.max(img_size) / 2)
##        p = p0 * 18 # Fullframe half size, 18mm
##        p = p0 * 18 / crop_factor # Fullframe half size, 18mm, with crop factor 1.5

        #Now, with the original point in the center of lens, the x,y,z is actually focal_length,x, y
        #therefore, theta = x/focal_length
        #phi = y/sqrt(x**2 + y**2 + focal_length **2)

        theta = np.arctan2(p[:, 0], focal_length)
        phi = np.arcsin(p[:, 1] / np.sqrt(np.sum(p ** 2, axis=1) + focal_length ** 2))
        return np.stack((theta, phi), axis=-1)

    @staticmethod
    def extract_point_features(sph, vol, k=15):
        logging.debug("extract_point_features()")
        pts_num = len(sph)
        vec = np.stack((np.cos(sph[:, 1]) * np.cos(sph[:, 0]),
                        np.cos(sph[:, 1]) * np.sin(sph[:, 0]),
                        np.sin(sph[:, 1])), axis=-1)
        dist_mat = 1 - spd.cdist(vec, vec, "cosine")
        vec_dist_ind = np.argsort(-dist_mat)
        dist_mat = np.where(dist_mat < -1, -1, np.where(dist_mat > 1, 1, dist_mat))
        dist_mat = np.arccos(dist_mat[np.array(range(pts_num))[:, np.newaxis], vec_dist_ind[:, :2 * k]])
        vol = vol[vec_dist_ind[:, :2 * k]]
        vol_ind = np.argsort(-vol * dist_mat)

        def make_cross_mat(v):
            return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

        theta_feature = np.zeros((pts_num, k))
        rho_feature = np.zeros((pts_num, k))
        vol_feature = np.zeros((pts_num, k))

        for i in range(pts_num):
            v0 = vec[i]
            vs = vec[vec_dist_ind[i, vol_ind[i, :k]]]
            angles = np.inner(vs, make_cross_mat(v0))
            angles = angles / la.norm(angles, axis=1)[:, np.newaxis]
            cr = np.inner(angles, make_cross_mat(angles[0]))
            s = la.norm(cr, axis=1) * np.sign(np.inner(cr, v0))
            c = np.inner(angles, angles[0])
            theta_feature[i] = np.arctan2(s, c)
            rho_feature[i] = dist_mat[i, vol_ind[i, :k]]
            vol_feature[i] = vol[i, vol_ind[i, :k]]

        fx = np.arange(-np.pi, np.pi, 3 * np.pi / 180)
        features = np.zeros((pts_num, len(fx)))
        for i in range(k):
            sigma = 2.5 * np.exp(-rho_feature[:, i] * 100) + .04
            tmp = np.exp(-np.subtract.outer(theta_feature[:, i], fx) ** 2 / 2 / sigma[:, np.newaxis] ** 2)
            tmp = tmp * (vol_feature[:, i] * rho_feature[:, i] ** 2 / sigma)[:, np.newaxis]
            features += tmp

        features = features / np.sqrt(np.sum(features ** 2, axis=1)).reshape((pts_num, 1))
        return features

    @staticmethod
    def find_initial_match(feature1, feature2):
        logging.debug("find_initial_match()")
        measure_dist_mat = spd.cdist(feature1["feature"], feature2["feature"], "cosine")
        pts1, pts2 = feature1["pts"], feature2["pts"]
        pts_mean = np.mean(np.vstack((pts1, pts2)), axis=0)
        pts_min = np.min(np.vstack((pts1, pts2)), axis=0)
        pts_max = np.max(np.vstack((pts1, pts2)), axis=0)
        pts_dist_mat = spd.cdist((pts1 - pts_mean) / (pts_max - pts_min), (pts2 - pts_mean) / (pts_max - pts_min),
                                 "euclidean")
        alpha = 0.00
        dist_mat = measure_dist_mat * (1 - alpha) + pts_dist_mat * alpha
        num1, num2 = dist_mat.shape

        # For a given point p1 in image1, find the most similar point p12 in image2,
        # then find the point p21 in image1 that most similar to p12, check the
        # distance between p1 and p21.

        idx12 = np.argsort(dist_mat, axis=1)
        idx21 = np.argsort(dist_mat, axis=0)
        ind = idx21[0, idx12[:, 0]] == range(num1)

        # Check Euclidean distance between the nearest pair
        d_th = min(np.percentile(dist_mat[range(num1), idx12[:, 0]], 30),
                   np.percentile(dist_mat[idx21[0, :], range(num2)], 30))
        ind = np.logical_and(ind, dist_mat[range(num1), idx12[:, 0]] < d_th)

        pair_idx = np.stack((np.where(ind)[0], idx12[ind, 0]), axis=-1)

        # Check angular distance between the nearest pair
        xyz1 = np.stack((np.cos(feature1["sph"][:, 1]) * np.cos(feature1["sph"][:, 0]),
                         np.cos(feature1["sph"][:, 1]) * np.sin(feature1["sph"][:, 0]),
                         np.sin(feature1["sph"][:, 1])), axis=-1)
        xyz2 = np.stack((np.cos(feature2["sph"][:, 1]) * np.cos(feature2["sph"][:, 0]),
                         np.cos(feature2["sph"][:, 1]) * np.sin(feature2["sph"][:, 0]),
                         np.sin(feature2["sph"][:, 1])), axis=-1)
        theta = np.arccos(np.sum(xyz1[pair_idx[:, 0]] * xyz2[pair_idx[:, 1]], axis=1))
        theta_th = min(np.percentile(theta, 75), np.pi / 6)

        pts_dist = la.norm(feature1["pts"][pair_idx[:, 0]] - feature2["pts"][pair_idx[:, 1]], axis=1)
        dist_th = max(np.max(feature1["pts"]), np.max(feature2["pts"])) * 0.3
        pair_idx = pair_idx[np.logical_and(theta < theta_th, pts_dist < dist_th)]
        
        logging.debug("find {0} pairs for initial".format(len(pair_idx)))
        return pair_idx

    @staticmethod
    def fine_tune_transform(feature1, feature2, init_pair_idx):
        ind = []
        k = 1
        while len(ind) < 0.6 * min(len(feature1["pts"]), len(feature2["pts"])) and k <= 10:
            if k==10:
                raise ValueError("Optimal alignment cannot be achieved.")
            # Step 1. Randomly choose 20 points evenly distributed on the image
            rand_pts = np.random.rand(20, 2) * (np.amax(feature1["pts"], axis=0) - np.amin(feature1["pts"], axis=0)) * \
                       np.array([1, 0.8]) + np.amin(feature1["pts"], axis=0)
            # Step 2. Find nearest points from feature1
            dist_mat = spd.cdist(rand_pts, feature1["pts"][init_pair_idx[:, 0]])
            tmp_ind = np.argmin(dist_mat, axis=1)
            # Step 3. Use these points to find a homography
            tf = cv2.findHomography(feature1["pts"][init_pair_idx[tmp_ind, 0]], feature2["pts"][init_pair_idx[tmp_ind, 1]],
                                    method=cv2.RANSAC, ransacReprojThreshold=5)

            # Then use the transform find more matched points
            pts12 = cv2.perspectiveTransform(np.array([[p] for p in feature1["pts"]], dtype="float32"), tf[0])[:, 0, :]
            dist_mat = spd.cdist(pts12, feature2["pts"])
            num1, num2 = dist_mat.shape

            idx12 = np.argsort(dist_mat, axis=1)
            tmp_ind = np.argwhere(np.array([dist_mat[i, idx12[i, 0]] for i in range(num1)]) < 5)
            if len(tmp_ind) > len(ind):
                ind = tmp_ind
            logging.debug("len(ind) = %d, len(feature) = %d", len(ind), min(len(feature1["pts"]), len(feature2["pts"])))
            k += 1

        pair_idx = np.hstack((ind, idx12[ind, 0]))

        tf = cv2.findHomography(feature1["pts"][pair_idx[:, 0]], feature2["pts"][pair_idx[:, 1]],
                                method=cv2.RANSAC, ransacReprojThreshold=5)
        return tf, pair_idx

    @staticmethod
    def convert_to_float(np_image):
        if np_image.dtype == np.float32 or np_image.dtype == np.float64:
            return np.copy(np_image)
        else:
            return np_image.astype("float32") / np.iinfo(np_image.dtype).max

    @staticmethod
    def read_tif_image(full_path):
        img = tiff.imread(full_path)
        exif_info = Tyf.open(full_path)
        return img, exif_info

    @staticmethod
    def read_raw_image(full_path):
        with rawpy.imread(full_path) as raw:
            img = raw.postprocess(use_camera_wb=True, output_bps=16)
        return img, None

    @staticmethod
    def save_tif_image(full_path, img, exif=None):
        if img.dtype != np.uint8 and img.dtype != np.uint16:
            return
        logging.debug("saving image...")
        tiff.imsave(full_path, img)
        # Not saving exif as it's problematic
        #tmp_exif = Tyf.open(full_path)
        #tmp_exif.load_raster()
        #if exif and isinstance(exif, Tyf.TiffFile):
        #   logging.debug("saving exif...")
        #    exif[0].stripes = tmp_exif[0].stripes
        #    exif.save(full_path)

