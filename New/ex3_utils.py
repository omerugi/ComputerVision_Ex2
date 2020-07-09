import cv2 as cv2
import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
import matplotlib.pyplot as plt

def SobelFlow(im1: np.ndarray, im2: np.ndarray):
    Iy = cv2.Sobel(im1, -1, 0, 1)
    Ix = cv2.Sobel(im1, -1, 1, 0)
    It = im1 - im2
    return Ix, Iy, It


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size: int, win_size: int) -> (np.ndarray, np.ndarray):
    """
Given two images, returns the Translation from im1 to im2
:param im1: Image 1
:param im2: Image 2
:param step_size: The image sample size:
:param win_size: The optical flow window size (odd number)
:return: Original points [[y,x]...], [[dU,dV]...] for each points
"""
    Ix, Iy, It = SobelFlow(im1, im2)
    q1 = []
    q2 = []
    for i in range(step_size, im1.shape[0], step_size):
        for j in range(step_size, im1.shape[1], step_size):
            try:
                ix = i - win_size // 2
                iy = i + 1 + win_size // 2
                iq = j - win_size // 2
                iz = j + 1 + win_size // 2
                winx2 = win_size * win_size
                windowIx = Ix[ix:iy, iq:iz]
                windowIy = Iy[ix:iy, iq:iz]
                windowIt = It[ix:iy, iq:iz]
                if windowIx.size < winx2:
                    break
                A = np.concatenate(
                    (windowIx.reshape((winx2, 1)), windowIy.reshape((winx2, 1))), axis=1)
                b = (windowIt.reshape((winx2, 1)))
                eig, _ = LA.eig(np.dot(A.T, A))
                eig = np.sort(eig)
                if eig[1] >= eig[0] > 1 and (eig[1] / eig[0]) < 100:
                    v = np.dot(np.dot(inv(np.dot(A.T, A)), A.T), b)
                    q1.append(np.array([j, i]))
                    q2.append(v)

            except IndexError as e:
                pass
    return np.array(q1), np.array(q2)


def fix_img(img, levels) -> np.ndarray:
    h = pow(2, levels) * (img.shape[0] // pow(2, levels))
    w = pow(2, levels) * (img.shape[1] // pow(2, levels))
    return img[:h, :w]


def gaussianPyr(img, levels=4) -> list:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    q = []
    img = fix_img(img, levels)
    q.append(img)
    I = img.copy()
    for i in range(1, levels):
        I = cv2.GaussianBlur(I, (5, 5), 1.1)
        I = I[::2, ::2]
        q.append(I)
        I = I.copy()
    return q


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up

    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """
    if len(img.shape) == 2:
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
    q = []
    kernel = cv2.getGaussianKernel(5, 1.1)
    img = fix_img(img, levels)
    imglist = gaussianPyr(img, levels)
    src = img.copy()
    for i in range(1, levels):
        new = gaussExpand(imglist[i], kernel * kernel.transpose() * 4)
        q.append(src - new)
        src = imglist[i]

    q.append(imglist[levels - 1])

    return q


def laplaceianExpand(lap_pyr: list) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    kernel = cv2.getGaussianKernel(5, 1.1)
    lap_pyr.reverse()

    base_img = lap_pyr[0]
    for i in range(1, len(lap_pyr)):
        ex_img = gaussExpand(base_img, kernel * kernel.transpose() * 4)
        base_img = ex_img + lap_pyr[i]

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

    kernel = cv2.getGaussianKernel(5, 1.1)
    img_1 = fix_img(img_1, levels)
    img_2 = fix_img(img_2, levels)
    mask = fix_img(mask, levels)

    naive = img_1 * mask + (1 - mask) * img_2

    qM = gaussianPyr(mask, levels)
    q1 = laplaceianReduce(img_1, levels)
    q2 = laplaceianReduce(img_2, levels)

    curr = q1[levels - 1] * qM[levels - 1] + (1 - qM[levels - 1]) * q2[levels - 1]

    for i in range(levels - 2, -1, -1):
        curr = gaussExpand(curr, kernel * kernel.transpose() * 4) + q1[i] * qM[i] + (1 - qM[i]) * q2[i]

    return naive, curr
    pass
