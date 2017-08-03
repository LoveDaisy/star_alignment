#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import tifffile as tiff
import numpy as np
import numpy.linalg as la
import scipy.spatial.distance as spd
import pywt
import piexif
import logging
# from matplotlib import pyplot as plt


def read_double_image(img_name):
    logging.debug("read_double_image()")
    img = tiff.imread(img_name)
    info = np.iinfo(img.dtype)
    exif_dict = piexif.load(img_name)

    focal_len = None
    for tag, v in exif_dict["Exif"].iteritems():
        if piexif.TAGS["Exif"][tag]["name"] == "FocalLength":
            focal_len = float(v[0]) / v[1]
            break
    return img.astype(np.float32) / info.max, focal_len


def save_double_image(img, img_name):
    logging.debug("save_double_image()")
    tmp_img = (img * np.iinfo(np.dtype("uint16")).max).astype(np.uint16)
    tiff.imsave(img_name, tmp_img)


def detect_star_point(img_gray):
    logging.debug("detect_star_point()")
    sigma = 3
    img_blr = cv2.GaussianBlur(img_gray, (9, 9), sigma)
    img_blr = (img_blr - np.mean(img_blr)) / (np.max(img_blr) - np.min(img_blr))

    coeffs = pywt.wavedec2(img_blr, "db8", level=6)
    coeffs[0].fill(0)
    coeffs[-1][0].fill(0)
    coeffs[-1][1].fill(0)
    coeffs[-1][2].fill(0)

    img_rec = pywt.waverec2(coeffs, "db8")
    bw = (img_rec > np.percentile(img_rec, 99.5)).astype(np.uint8) * 255

    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(np.copy(bw), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = filter(lambda x: len(x) > 5, contours)

    elps = map(cv2.fitEllipse, contours)
    centroids = np.array(map(lambda e: e[0], elps))
    # centroids = np.array(map(lambda pts: np.sum(l1_norm(np.array(map(
    #     lambda ind: [[img_gray[ind[0][1]][ind[0][0]]]], pts))) * pts, axis=0)[0], contours))
    areas = np.array(map(lambda x: cv2.contourArea(x) + 0.5 * len(x), contours))
    eccentricities = np.sqrt(np.array(map(lambda x: 1-(x[1][0] / x[1][1])**2, elps)))

    mask = np.zeros(bw.shape, np.uint8)
    intensities = np.zeros(areas.shape)
    for i in range(len(contours)):
        cv2.drawContours(mask, contours[i], 0, 255, -1)
        rect = cv2.boundingRect(contours[i])
        val = cv2.mean(img_rec[rect[1]:rect[1]+rect[3]+1, rect[0]:rect[0]+rect[2]+1],
                       mask[rect[1]:rect[1]+rect[3]+1, rect[0]:rect[0]+rect[2]+1])
        mask[rect[1]:rect[1]+rect[3]+1, rect[0]:rect[0]+rect[2]+1] = 0
        intensities[i] = val[0]
    # intensities = np.array(map(lambda c: img_gray[int(c[1]), int(c[0])], centroids))

    inds = np.logical_and(areas > 5, areas < 200, eccentricities < .9)
    inds = np.logical_and(inds, areas > np.percentile(areas, 20), intensities > np.percentile(intensities, 20))

    star_pts = centroids[inds]      # [x, y]
    areas = areas[inds]
    intensities = intensities[inds]

    return star_pts, areas * intensities


def convert_coord_img_sph(star_pts, img_size, f):
    logging.debug("convert_coord_img_sph()")
    p0 = (star_pts - img_size / 2.0) / (np.max(img_size) / 2)
    p = p0 * 18     # Fullframe half size, 18mm
    lam = np.arctan2(p[:, 0], f)
    phi = np.arcsin(p[:, 1] / np.sqrt(np.sum(p ** 2, axis=1) + f ** 2))
    return np.stack((lam, phi), axis=-1)


def extract_point_features(sph, vol, k=15):
    logging.debug("extract_point_features()")
    pts_num = len(sph)
    vec = np.stack((np.cos(sph[:, 1]) * np.cos(sph[:, 0]),
                    np.cos(sph[:, 1]) * np.sin(sph[:, 0]),
                    np.sin(sph[:, 1])), axis=-1)
    dist_mat = 1 - spd.cdist(vec, vec, "cosine")
    vec_dist_ind = np.argsort(-dist_mat)
    dist_mat = np.where(dist_mat < -1, -1, np.where(dist_mat > 1, 1, dist_mat))
    dist_mat = np.arccos(dist_mat[np.array(range(pts_num))[:, np.newaxis], vec_dist_ind[:, :2*k]])
    vol = vol[vec_dist_ind[:, :2*k]]
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

    fx = np.arange(-np.pi, np.pi, 3*np.pi/180)
    features = np.zeros((pts_num, len(fx)))
    for i in range(k):
        sigma = 2.5 * np.exp(-rho_feature[:, i] * 100) + .04
        tmp = np.exp(-np.subtract.outer(theta_feature[:, i], fx) ** 2 / 2 / sigma[:, np.newaxis] ** 2)
        tmp = tmp * (vol_feature[:, i] * rho_feature[:, i] ** 2 / sigma)[:, np.newaxis]
        features += tmp

    features = features / np.sqrt(np.sum(features ** 2, axis=1)).reshape((pts_num, 1))
    return features


def find_initial_match(feature1, feature2):
    logging.debug("find_initial_match()")
    measure_dist_mat = spd.cdist(feature1["feature"], feature2["feature"], "cosine")
    pts1, pts2 = feature1["pts"], feature2["pts"]
    pts_mean = np.mean(np.vstack((pts1, pts2)), axis=0)
    pts_min = np.min(np.vstack((pts1, pts2)), axis=0)
    pts_max = np.max(np.vstack((pts1, pts2)), axis=0)
    pts_dist_mat = spd.cdist((pts1 - pts_mean) / (pts_max - pts_min), (pts2 - pts_mean) / (pts_max - pts_min), "euclidean")
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

    # pts12 = cv2.perspectiveTransform(np.array([[p] for p in feature1["pts"]], dtype="float32"), tf[0])[:, 0, :]
    # dist_mat = spd.cdist(pts12, feature2["pts"])
    # num1, num2 = dist_mat.shape
    #
    # idx12 = np.argsort(dist_mat, axis=1)
    # ind = np.argwhere(np.array([dist_mat[i, idx12[i, 0]] for i in range(num1)]) < 5)
    # pair_idx = np.hstack((ind, idx12[ind, 0]))
    #
    # tf = cv2.findHomography(feature1["pts"][pair_idx[:, 0]], feature2["pts"][pair_idx[:, 1]],
    #                         method=cv2.RANSAC, ransacReprojThreshold=5)
    return tf, pair_idx


def align_average(img_store, feature_store, ref_index=0):
    logging.debug("align_average()")
    mean_img = np.copy(img_store[ref_index])
    img_size = mean_img.shape

    feature_ref = feature_store[ref_index]
    img_num = 1
    for i in range(len(img_store)):
        if i == ref_index:
            continue

        img_num += 1
        img = img_store[i]
        feature = feature_store[i]

        pair_idx = find_initial_match(feature, feature_ref)

        # plt.imshow(img)
        # plt.plot(feature["pts"][pair_idx[:,0],0], feature["pts"][pair_idx[:,0],1], 'yx')
        # plt.plot(feature_ref["pts"][pair_idx[:,1],0], feature_ref["pts"][pair_idx[:,1],1], 'm+')
        # plt.show()

        tf, pair_idx = fine_tune_transform(feature, feature_ref, pair_idx)

        # plt.imshow(img)
        # plt.plot(feature["pts"][pair_idx[:, 0], 0], feature["pts"][pair_idx[:, 0], 1], 'yx')
        # plt.plot(feature_ref["pts"][pair_idx[:, 1], 0], feature_ref["pts"][pair_idx[:, 1], 1], 'm+')
        # plt.show()

        img_tf = cv2.warpPerspective(img, tf[0], (img_size[1], img_size[0]))
        mean_img = mean_img / img_num * (img_num - 1) + img_tf / img_num

    return mean_img


if __name__ == "__main__":
    logging_level = logging.DEBUG
    logging_format = "%(asctime)s (%(name)s) [%(levelname)s] line %(lineno)d: %(message)s"
    logging.basicConfig(format=logging_format, level=logging_level)

    img_path = u"/Users/jiajiezhang/Desktop/给八爪/"

    img_store = []
    feature_store = []

    # img_names = ["DSC_{:04d}.tif".format(i) for i in range(2955, 2957)]
    img_names = [u"{:s}IMG_{:04d}.tif".format(img_path, i) for i in [2323, 2355]]
    for img_name in img_names:
        print("Reading images...")
        img, focal_len = read_double_image(img_name)
        if not focal_len:
            focal_len = 20
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_store.append(img)

        print("Detecting stars...")
        star_pts, vol = detect_star_point(img_gray)

        print("Extracting star point features...")
        sph = convert_coord_img_sph(star_pts, np.array(img_gray.shape)[::-1], focal_len)
        features = extract_point_features(sph, vol)
        feature_store.append({"feature": features, "pts": star_pts, "vol": vol, "sph": sph})

        # plt.imshow(img_gray)
        # plt.scatter(star_pts[:, 0], star_pts[:, 1], c="k", marker="x")
        # plt.show()

    img = align_average(img_store, feature_store)

    # plt.imshow(img)
    # plt.show()


