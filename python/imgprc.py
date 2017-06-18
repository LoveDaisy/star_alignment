#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import numpy.linalg as la
import scipy.spatial.distance as spd
import pywt
import piexif
from matplotlib import pyplot as plt


def read_double_image(img_name):
    img = cv2.cvtColor(cv2.imread(img_name, cv2.CV_LOAD_IMAGE_UNCHANGED), cv2.COLOR_BGR2RGB)
    info = np.iinfo(img.dtype)
    exif_dict = piexif.load(img_name)

    focal_len = None
    for tag, v in exif_dict["Exif"].iteritems():
        if piexif.TAGS["Exif"][tag]["name"] == "FocalLength":
            focal_len = float(v[0]) / v[1]
            break
    return img.astype(np.float32) / info.max, focal_len


def detect_star_point(img_gray):
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

    def l1_norm(v):
        return v / np.sum(v)

    elps = map(cv2.fitEllipse, contours)
    centroids = np.array(map(lambda pts: np.sum(l1_norm(np.array(map(
        lambda ind: [[img_gray[ind[0][1]][ind[0][0]]]], pts))) * pts, axis=0)[0], contours))
    areas = np.array(map(len, contours))
    eccentricities = np.sqrt(np.array(map(lambda x: 1-(x[1][0] / x[1][1])**2, elps)))
    intensities = np.array(map(lambda pts: np.mean(np.array(
        map(lambda ind: img_gray[ind[0][1]][ind[0][0]], pts))), contours))

    inds = np.logical_and(areas > 5, areas < 100, eccentricities < .9)
    inds = np.logical_and(inds, areas > np.percentile(areas, 20), intensities > np.percentile(intensities, 20))
    star_pts = centroids[inds]      # [x, y]

    return star_pts, areas * intensities


def convert_coord_img_sph(star_pts, img_size, f):
    p0 = (star_pts - img_size / 2.0) / (np.max(img_size) / 2)
    p = p0 * 18     # Fullframe half size, 18mm
    lam = np.arctan2(p[:, 0], f)
    phi = np.arcsin(p[:, 1] / np.sqrt(np.sum(p ** 2, axis=1) + f ** 2))
    return np.stack((lam, phi), axis=-1)


def extract_point_features(sph, vol, k=15):
    pts_num = len(sph)
    vec = np.stack((np.cos(sph[:, 1]) * np.cos(sph[:, 0]),
                    np.cos(sph[:, 1]) * np.sin(sph[:, 0]),
                    np.sin(sph[:, 1])), axis=-1)
    dist_mat = 1 - spd.cdist(vec, vec, "cosine")
    vec_dist_ind = np.argsort(-dist_mat)
    dist_mat = np.where(dist_mat < -1, -1, np.where(dist_mat > 1, 1, dist_mat))
    dist_mat = np.arccos(dist_mat[np.array(range(pts_num))[:, np.newaxis], vec_dist_ind[:, :2*k]])
    vol = vol[vec_dist_ind[:, 1:2*k]]
    vol_ind = np.argsort(-vol * dist_mat[:, 1:])

    def make_cross_mat(v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    theta_feature = np.zeros((pts_num, k))
    rho_feature = np.zeros((pts_num, k))
    vol_feature = np.zeros((pts_num, k))

    for i in range(pts_num):
        v0 = vec[i]
        vs = vec[vec_dist_ind[i, vol_ind[i, :k]+1]]
        angles = np.inner(vs, make_cross_mat(v0))
        angles = angles / la.norm(angles, axis=1)[:, np.newaxis]
        cr = np.inner(angles, make_cross_mat(angles[0]))
        s = la.norm(cr, axis=1) * np.sign(np.inner(cr, v0))
        c = np.inner(angles, angles[0])
        theta_feature[i] = np.arctan2(s, c)
        rho_feature[i] = dist_mat[i, vol_ind[i, :k]+1]
        vol_feature[i] = vol[i, vol_ind[i, :k]]

    fx = np.arange(-np.pi, np.pi, 3*np.pi/180)
    features = np.zeros((pts_num, len(fx)))
    for i in range(k):
        sigma = 2.5 * np.exp(-rho_feature[:, i] * 100) + .04
        tmp = np.exp(-np.subtract.outer(theta_feature[:, i], fx) ** 2 / 2 / sigma[:, np.newaxis] ** 2)
        tmp = tmp * (vol_feature[:, i] * rho_feature[:, i] ** 2 / sigma)[:, np.newaxis]
        features += tmp

    return features


def find_initial_match(feature1, feature2):
    dist_mat = spd.cdist(feature1["feature"], feature2["feature"], "cosine")
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
    return pair_idx


def align_average(img_store, feature_store, ref_index=0):
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
        tf = cv2.findHomography(feature["pts"][pair_idx[:, 0]], feature_ref["pts"][pair_idx[:, 1]],
                                method=cv2.RANSAC, ransacReprojThreshold=5)

        img_tf = cv2.warpPerspective(img, tf[0], (img_size[1], img_size[0]))
        mean_img = mean_img / img_num * (img_num - 1) + img_tf / img_num

    return mean_img


if __name__ == "__main__":
    img_path = "./"

    img_store = []
    feature_store = []

    for img_name in ["IMG_{:04d}_0.TIF".format(590), "IMG_{:04d}_0.TIF".format(585)]:
        print("Reading images...")
        img, focal_len = read_double_image(img_path + img_name)
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

    plt.imshow(img)
    plt.show()


