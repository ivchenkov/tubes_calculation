import numpy as np
import cv2

ZMPP = 0.7676


def cnts_mean(img, cnts):
    img = img.copy()
    mask = np.zeros(img.shape, img.dtype)
    cv2.fillPoly(mask, pts =cnts, color=1)
    masked_img = img * mask
    areas = [cv2.contourArea(cnt) for cnt in cnts]
    area = np.array(areas).sum()
    return masked_img.sum() / area


def cnts_sum(img, cnts):
    img = img.copy()
    mask = np.zeros(img.shape, img.dtype)
    cv2.fillPoly(mask, pts = cnts, color=1)
    masked_img = img * mask
    return masked_img.sum()


def cnt_lateral_length(cnt, n_iteration=10):
    cnt_area = cv2.contourArea(cnt)
    cnt_perimeter = cv2.arcLength(cnt, True)
    l = cnt_perimeter / 2
    for _ in range(n_iteration): 
        dl = cnt_area / l
        l = cnt_perimeter / 2 - dl
    return l


def mb_surface_intensity(img, mb_cnts, bg_cnts):
    img = img.copy()
    bg = cnts_mean(img, bg_cnts)
    i_m = cnts_sum(img - bg, mb_cnts)
    l = sum([cnt_lateral_length(cnt) for cnt in mb_cnts])
    return i_m / (l * ZMPP )


def tb_pure_intensity(img, t_cnt, bg_cnt):
    bg = cnts_mean(img, [bg_cnt])
    i_t = cnts_sum(img - bg, [t_cnt])
    return i_t