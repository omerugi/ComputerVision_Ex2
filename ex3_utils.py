import math
from math import exp

import cv2 as cv2
import numpy as np
from matplotlib.mathtext import List
from numpy.linalg import inv
from numpy import linalg as LA
import matplotlib.pyplot as plt


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size: int, win_size: int) -> (np.ndarray, np.ndarray):
    """
Given two images, returns the Translation from im1 to im2
:param im1: Image 1
:param im2: Image 2
:param step_size: The image sample size:
:param win_size: The optical flow window size (odd number)
:return: Original points [[y,x]...], [[dU,dV]...] for each points
"""
    Iy = cv2.Sobel(im1, -1, 0, 1)
    Ix = cv2.Sobel(im1, -1, 1, 0)
    It = im1 - im2

    res1 = []
    res2 = []
    for i in range(step_size, im1.shape[0], step_size):
        for j in range(step_size, im1.shape[1], step_size):
            try:
                windowIx = Ix[i - win_size // 2:i + 1 + win_size // 2, j - win_size // 2: j + 1 + win_size // 2]
                windowIy = Iy[i - win_size // 2:i + 1 + win_size // 2, j - win_size // 2: j + 1 + win_size // 2]
                windowIt = It[i - win_size // 2:i + 1 + win_size // 2, j - win_size // 2: j + 1 + win_size // 2]
                if windowIx.size < win_size * win_size:
                    break
                A = np.concatenate(
                    (windowIx.reshape((win_size * win_size, 1)), windowIy.reshape((win_size * win_size, 1))), axis=1)
                b = (windowIt.reshape((win_size * win_size, 1)))
                g, _ = LA.eig(np.dot(A.T, A))
                g=np.sort(g)
                print(g)
                if (g[1] >= g[0] > 1 and (g[1] / g[0]) < 100) :
                    v = np.dot(np.dot(inv(np.dot(A.T, A)), A.T), b)
                    res1.append(np.array([j, i]))
                    res2.append(v)
                    print(g[0],g[1])

            except IndexError as e:
                pass
    return np.array(res1), np.array(res2)


def blurImage2(in_image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
    """
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    guassian = cv2.getGaussianKernel(kernel_size, sigma)
    guassian = guassian * guassian.transpose()
    return cv2.filter2D(in_image, -1, guassian, borderType=cv2.BORDER_REPLICATE)


def gaussianPyr(img, levels=4) -> list:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    res = []
    h = pow(2, levels) * (img.shape[0] // pow(2, levels))
    w = pow(2, levels) * (img.shape[1] // pow(2, levels))
    img = img[:h, :w]
    res.append(img)
    I = img.copy()
    for i in range(1, levels):
        I = blurImage2(I, 5)
        I = I[::2, ::2]
        res.append(I)
        I = I.copy()
    return res


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up

    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """
    if (len(img.shape) == 2):
        out = np.zeros((2 * img.shape[0], 2 * img.shape[1]), dtype=img.dtype)
    else:
        out = np.zeros((2 * img.shape[0], 2 * img.shape[1], img.shape[2]), dtype=img.dtype)
    out[::2, ::2] = img
    return cv2.filter2D(out, -1, gs_k, borderType=cv2.BORDER_REPLICATE)


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> list:
    """
Creates a Laplacian pyramid
:param img: Original image
:param levels: Pyramid depth
:return: Laplacian Pyramid (list of images)
"""
    res = []
    sigma = 0.3 * ((5 - 1) * 0.5 - 1) + 0.8
    guassian = cv2.getGaussianKernel(5, sigma)
    guassian = guassian * guassian.transpose() * 4
    h = pow(2, levels) * (img.shape[0] // pow(2, levels))
    w = pow(2, levels) * (img.shape[1] // pow(2, levels))
    img = img[:h, :w]
    imglist = gaussianPyr(img, levels)
    orig = img.copy()
    for i in range(1, levels):
        exp = gaussExpand(imglist[i], guassian)
        res.append(orig - exp)
        # plt.imshow(orig-exp)
        # plt.show()
        orig = imglist[i]

    res.append(imglist[levels - 1])

    return res


def laplaceianExpand(lap_pyr: list) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    lap_pyr.reverse()

    temp = lap_pyr.pop(0)
    base_img = temp
    sigma = 0.3 * ((5 - 1) * 0.5 - 1) + 0.8
    guassian = cv2.getGaussianKernel(5, sigma)
    guassian = guassian * guassian.transpose() * 4

    for lap_img in lap_pyr:
        ex_img = gaussExpand(base_img, guassian)
        base_img = ex_img + lap_img

    lap_pyr.insert(0, temp)
    lap_pyr.reverse()
    return base_img


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: Blended Image
    """

    sigma = 0.3 * ((5 - 1) * 0.5 - 1) + 0.8
    guassian = cv2.getGaussianKernel(5, sigma)
    guassian = guassian * guassian.transpose() * 4

    h = pow(2, levels) * (img_1.shape[0] // pow(2, levels))
    w = pow(2, levels) * (img_1.shape[1] // pow(2, levels))
    img_1 = img_1[:h, :w]

    h = pow(2, levels) * (img_2.shape[0] // pow(2, levels))
    w = pow(2, levels) * (img_2.shape[1] // pow(2, levels))
    img_2 = img_2[:h, :w]

    h = pow(2, levels) * (mask.shape[0] // pow(2, levels))
    w = pow(2, levels) * (mask.shape[1] // pow(2, levels))
    mask = mask[:h, :w]



    list_mask = gaussianPyr(mask, levels)
    list_img_1 = laplaceianReduce(img_1, levels)
    list_img_2 = laplaceianReduce(img_2, levels)

    curr = list_img_1[levels - 1] * list_mask[levels - 1] + (1 - list_mask[levels - 1]) * list_img_2[levels - 1]

    for i in range(levels - 2, -1, -1):
        curr = gaussExpand(curr, guassian) + list_img_1[i] * list_mask[i] + (1 - list_mask[i]) * list_img_2[i]

    naive = img_1 * mask + (1 - mask) * img_2

    return naive, curr
    pass