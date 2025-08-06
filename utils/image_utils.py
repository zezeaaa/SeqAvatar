#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from PIL import Image
import imageio
import cv2

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def load_image_mask(image_path, mask_path, white_background, image_scaling=1.0, K=None, D=None):
    image = np.array(imageio.imread(image_path)/255., dtype=np.float32)
    bkgd_mask = imageio.imread(mask_path)
    if bkgd_mask.ndim == 3:
        bkgd_mask = bkgd_mask[:, :, 0]
    bkgd_mask = (bkgd_mask != 0).astype(np.uint8)
    if np.all(bkgd_mask == False):
        return image, None, None, K
    
    if D is not None and K is not None:
        image = cv2.undistort(image, K, D)
        bkgd_mask = cv2.undistort(bkgd_mask, K, D)
    image[bkgd_mask == 0] = 1 if white_background else 0
    
    if image_scaling != 1.:
        H, W = int(image.shape[0] * image_scaling), int(image.shape[1] * image_scaling)
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
        bkgd_mask = cv2.resize(bkgd_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        K[:2] = K[:2] * image_scaling
    
    bound_mask = get_rect_bound_mask(bkgd_mask)
    image = Image.fromarray(np.array(image*255.0, dtype=np.uint8), "RGB")
    bkgd_mask = Image.fromarray(np.array(bkgd_mask*255.0, dtype=np.uint8))

    return image, bkgd_mask, bound_mask, K

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1) # 4,5,7,6,4
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

def get_bound_3d(points, expand=0.05):
    min_xyz = np.min(points, axis=0) - expand
    max_xyz = np.max(points, axis=0) + expand
    bound = np.stack([min_xyz, max_xyz], axis=0)
    return bound

def get_rect_bound_mask(msk, expand_pixels=5):
    y_indices, x_indices = np.where(msk != 0)
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    y_min = max(y_min - expand_pixels, 0)
    y_max = min(y_max + expand_pixels, msk.shape[0] - 1)
    x_min = max(x_min - expand_pixels, 0)
    x_max = min(x_max + expand_pixels, msk.shape[1] - 1)

    bound_mask = np.zeros_like(msk)
    bound_mask[y_min:y_max+1, x_min:x_max+1] = 1
    bound_mask = Image.fromarray(np.array(bound_mask*255.0, dtype=np.uint8))
    return bound_mask