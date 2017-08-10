#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Author: Jiajie Zhang
Email: zhangjiajie043@gmail.com
"""

import logging
import os
import cv2
import numpy as np
import numpy.linalg as la
import scipy.spatial.distance as spd
import pywt
import Tyf
import tifffile as tiff


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
        _, ext = os.path.splitext(full_path)
        if ext.lower() in (".tiff", ".tif") and os.path.isfile(full_path):
            self.original_image, self.exif_info = ImageProcessing.read_tif_image(full_path)
            gray_img = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
            self.fullsize_gray_image = gray_img.astype("float32") / np.iinfo(gray_img.dtype).max
            self.reset_focal_length()
        else:
            self.original_image = None
            self.fullsize_gray_image = None
            self.exif_info = None
            self.focal_len = None

        self.features = {}

    def reset_focal_length(self):
        f = self.get_exif_value("FocalLength")
        if f and len(f) == 2:
            self.focal_len = f[0] * 1.0 / f[1]
        elif f and len(f) == 1:
            self.focal_len = f[0]
        else:
            self.focal_len = None

    def get_exif_value(self, name):
        if not self.exif_info:
            return None

        info = self.exif_info[0].find(name)
        if not info:
            return None
        else:
            return info.value


class DataModel(object):
    def __init__(self):
        super(DataModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.images = []
        self.ref_ind = 0
        self.image_dir = None
        self.final_img = None  # Of type double

        # On concurrent
        self.is_adding_image = False

    def add_image(self, path):
        self.logger.debug("add_image()")
        img_dir, name = os.path.split(path)

        if not os.path.exists(path) or not os.path.isfile(path):
            self.logger.error("File %s not exists!", path)
            return False

        for img in self.images:
            if path == img.full_path:
                self.logger.info("Image is already open. File: %s", path)
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

    def clear_images(self):
        self.logger.debug("clear_images()")
        self.images = []
        self.final_img = None
        self.image_dir = None
        self.ref_ind = 0

        self.is_adding_image = False

    def reset_result(self):
        self.logger.debug("reset_result()")
        self.final_img = None
        for img in self.images:
            img.features = {}

    def has_image(self):
        self.logger.debug("has_image()")
        return len(self.images) > 0

    def iter_images(self):
        self.logger.debug("iter_images()")
        return iter(self.images)

    def total_images(self):
        self.logger.debug("total_images()")
        return len(self.images)

    def has_result(self):
        self.logger.debug("has_result()")
        return self.final_img is not None


class ImageProcessing(object):
    def __init__(self):
        super(ImageProcessing, self).__init__()

    @staticmethod
    def _try_wavedec(img_blr, resize_factor=0.25):
        img_shape = img_blr.shape

        need_resize = abs(resize_factor - 1) > 0.001

        if need_resize:
            img_blr_resize = cv2.resize(img_blr, None, fx=resize_factor, fy=resize_factor)
        else:
            img_blr_resize = img_blr
        coeffs = pywt.wavedec2(img_blr_resize, "db8", level=6)

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
    def detect_star_points(img_gray):
        logging.debug("detect_star_point()")
        logging.debug("img type: %s", img_gray.dtype)

        sigma = 3
        img_shape = img_gray.shape
        img_blr = cv2.GaussianBlur(img_gray, (9, 9), sigma)
        logging.debug("img_blr type: %s", img_blr.dtype)
        img_blr = (img_blr - np.mean(img_blr)) / (np.max(img_blr) - np.min(img_blr))

        resize_factor = 1
        while max(img_shape) * resize_factor > 2200:
            resize_factor *= 0.5
        while True:
            try:
                img_rec = ImageProcessing._try_wavedec(img_blr, resize_factor=resize_factor)

                bw = (img_rec > np.percentile(img_rec, 99.5)).astype(np.uint8) * 255
                bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

                contours, _ = cv2.findContours(np.copy(bw), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                contours = filter(lambda x: len(x) > 5, contours)
                logging.debug("%d star points detected", len(contours))

                if len(contours) > 200:
                    break
                else:
                    raise ValueError, "No enough points"
            except ValueError:
                if resize_factor >= 1:
                    raise (ValueError, "Cannot detect enough star points")
                else:
                    resize_factor *= 2
        logging.debug("resize factor = %f", resize_factor)

        elps = map(cv2.fitEllipse, contours)
        centroids = np.array(map(lambda e: e[0], elps))
        areas = np.array(map(lambda x: cv2.contourArea(x) + 0.5 * len(x), contours))
        eccentricities = np.sqrt(np.array(map(lambda x: 1 - (x[1][0] / x[1][1]) ** 2, elps)))

        mask = np.zeros(bw.shape, np.uint8)
        intensities = np.zeros(areas.shape)
        for i in range(len(contours)):
            cv2.drawContours(mask, contours[i], 0, 255, -1)
            rect = cv2.boundingRect(contours[i])
            val = cv2.mean(img_rec[rect[1]:rect[1] + rect[3] + 1, rect[0]:rect[0] + rect[2] + 1],
                           mask[rect[1]:rect[1] + rect[3] + 1, rect[0]:rect[0] + rect[2] + 1])
            mask[rect[1]:rect[1] + rect[3] + 1, rect[0]:rect[0] + rect[2] + 1] = 0
            intensities[i] = val[0]

        inds = np.logical_and(areas > 5, areas < 200, eccentricities < .9)
        inds = np.logical_and(inds, areas > np.percentile(areas, 20), intensities > np.percentile(intensities, 20))

        star_pts = centroids[inds]  # [x, y]
        areas = areas[inds]
        intensities = intensities[inds]

        return star_pts, areas * intensities

    @staticmethod
    def convert_to_spherical_coord(star_pts, img_size, f):
        logging.debug("convert_coord_img_sph()")
        p0 = (star_pts - img_size / 2.0) / (np.max(img_size) / 2)
        p = p0 * 18  # Fullframe half size, 18mm
        lam = np.arctan2(p[:, 0], f)
        phi = np.arcsin(p[:, 1] / np.sqrt(np.sum(p ** 2, axis=1) + f ** 2))
        return np.stack((lam, phi), axis=-1)

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

        # plt.plot(feature1["pts"][:, 0], feature1["pts"][:, 1], 'x')
        # plt.plot(feature2["pts"][:, 0], feature2["pts"][:, 1], '+')
        # for p in pair_idx:
        #     plt.plot([feature1["pts"][p[0], 0], feature2["pts"][p[1], 0]],
        #              [feature1["pts"][p[0], 1], feature2["pts"][p[1], 1]])
        # plt.show()

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

        # plt.plot(feature1["pts"][:, 0], feature1["pts"][:, 1], 'x')
        # plt.plot(feature2["pts"][:, 0], feature2["pts"][:, 1], '+')
        # for p in pair_idx:
        #     plt.plot([feature1["pts"][p[0], 0], feature2["pts"][p[1], 0]],
        #              [feature1["pts"][p[0], 1], feature2["pts"][p[1], 1]])
        # plt.show()
        logging.debug("find {0} pairs for initial".format(len(pair_idx)))
        return pair_idx

    @staticmethod
    def fine_tune_transform(feature1, feature2, init_pair_idx):
        tf = cv2.findHomography(feature1["pts"][init_pair_idx[:, 0]], feature2["pts"][init_pair_idx[:, 1]],
                                method=cv2.RANSAC, ransacReprojThreshold=5)
        pts12 = cv2.perspectiveTransform(np.array([[p] for p in feature1["pts"]], dtype="float32"), tf[0])[:, 0, :]
        dist_mat = spd.cdist(pts12, feature2["pts"])
        num1, num2 = dist_mat.shape

        idx12 = np.argsort(dist_mat, axis=1)
        ind = np.argwhere(np.array([dist_mat[i, idx12[i, 0]] for i in range(num1)]) < 5)
        pair_idx = np.hstack((ind, idx12[ind, 0]))

        tf = cv2.findHomography(feature1["pts"][pair_idx[:, 0]], feature2["pts"][pair_idx[:, 1]],
                                method=cv2.RANSAC, ransacReprojThreshold=5)
        return tf, pair_idx

    @staticmethod
    def read_tif_image(full_path):
        img = tiff.imread(full_path)
        exif_info = Tyf.open(full_path)
        return img, exif_info

    @staticmethod
    def save_tif_image(full_path, img, exif=None):
        if img.dtype != np.uint8 and img.dtype != np.uint16:
            return
        logging.debug("saving image...")
        tiff.imsave(full_path, img)
        tmp_exif = Tyf.open(full_path)
        tmp_exif.load_raster()
        if exif and isinstance(exif, Tyf.TiffFile):
            logging.debug("saving exif...")
            exif[0].stripes = tmp_exif[0].stripes
            exif.save(full_path)

if __name__ == "__main__":
    logging_level = logging.DEBUG
    logging_format = "%(asctime)s (%(name)s) [%(levelname)s] line %(lineno)d: %(message)s"
    logging.basicConfig(format=logging_format, level=logging_level)

    img_tmpl = u"./IMG_{:04d}_0.tif"
    data_model = DataModel()
    for p in [img_tmpl.format(i) for i in (585, 590)]:
        logging.debug("image: %s", p)
        data_model.add_image(p)
    
    ref_img = data_model.images[0]
    f = ref_img.focal_len
    if not f:
        f = 50
    img_shape = ref_img.fullsize_gray_image.shape
    img_size = np.array([img_shape[1], img_shape[0]])
    data_model.reset_result()

    pts, vol = ImageProcessing.detect_star_points(ref_img.fullsize_gray_image)
    sph = ImageProcessing.convert_to_spherical_coord(pts, np.array((img_shape[1], img_shape[0])), f)
    feature = ImageProcessing.extract_point_features(sph, vol)
    ref_img.features["pts"] = pts
    ref_img.features["sph"] = sph
    ref_img.features["vol"] = vol
    ref_img.features["feature"] = feature

    data_model.final_img = np.copy(ref_img.original_image).astype("float32") / np.iinfo(
        ref_img.original_image.dtype).max

    img = data_model.images[1]
    pts, vol = ImageProcessing.detect_star_points(img.fullsize_gray_image)
    sph = ImageProcessing.convert_to_spherical_coord(pts, img_size, f)
    feature = ImageProcessing.extract_point_features(sph, vol)
    img.features["pts"] = pts
    img.features["sph"] = sph
    img.features["vol"] = vol
    img.features["feature"] = feature

    pair_idx = ImageProcessing.find_initial_match(img.features, ref_img.features)
    tf, pair_idx = ImageProcessing.fine_tune_transform(img.features, ref_img.features, pair_idx)
    img_tf = cv2.warpPerspective(img.original_image, tf[0], tuple(img_size))
    img_tf = img_tf.astype("float32") / np.iinfo(img_tf.dtype).max

    data_model.final_img = data_model.final_img / 2  + img_tf / 2
    result_img = (data_model.final_img * np.iinfo("uint16").max).astype("uint16")
    ImageProcessing.save_tif_image("test.tif", result_img, data_model.images[0].exif_info)

