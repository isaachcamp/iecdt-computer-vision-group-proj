
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml
import cv2

from fisheye_to_perspective_utils import extract_cube_map

# Get paths
A_DPATH = Path('/gws/nopw/j04/iecdt/computer-vision-data/cam_a')
B_DPATH = Path('/gws/nopw/j04/iecdt/computer-vision-data/cam_b')


def orientate_cameras():
    # Get list of fisheye image file names from absolute paths
    A_filenames = [item.name for item in list((A_DPATH / 'imgs').glob('*.png'))]
    B_filenames = [item.name for item in list((B_DPATH / 'imgs').glob('*.png'))]

    # Get common file names (same time stamp)
    common = [item for item in A_filenames if item in B_filenames]

    # Extract camera parameters
    with open("/gws/nopw/j04/iecdt/computer-vision-data/fisheye_calib_A.yml", 'r') as f:
        calib_A = yaml.safe_load(f)

    with open("/gws/nopw/j04/iecdt/computer-vision-data/fisheye_calib_B.yml", 'r') as f:
        calib_B = yaml.safe_load(f)

    # Get rotations for aligning cameras.
    transform = np.loadtxt('/gws/nopw/j04/iecdt/computer-vision-data/T_rel.txt')
    rotate_angle_A = np.rad2deg(np.arctan(transform[0,-1]/transform[1,-1]))
    rotate_angle_B = np.rad2deg(np.arccos(transform[0,0]))

    # Process images and save
    for fname in common:
        opath_img_a = A_DPATH / 'rectified_imgs' / fname
        opath_img_b = B_DPATH / 'rectified_imgs' / fname

        # Create directory if needed
        if not opath_img_a.parent.exists():
            opath_img_a.parent.mkdir(exist_ok=True, parents=True)
        if not opath_img_b.parent.exists():
            opath_img_b.parent.mkdir(exist_ok=True, parents=True)

        if opath_img_b.exists() and opath_img_a.exists():
            continue

        print('creating rectified images for:', fname)

        img_a = plt.imread(A_DPATH / 'imgs' / fname)
        img_b = plt.imread(B_DPATH / 'imgs' / fname)

        # Rectify fisheye images
        rectified_imgs_a = extract_cube_map(img_a, calib_A, rotate=rotate_angle_A)
        rectified_imgs_b = extract_cube_map(img_b, calib_B, rotate=rotate_angle_B+rotate_angle_A)

        # Save only the vertical rectified image
        plt.imsave(opath_img_a, rectified_imgs_a[-1])
        plt.imsave(opath_img_b, rectified_imgs_b[-1])


def compress_imgs():
    # Get list of rectified image file names
    A_filenames = [item.name for item in list((A_DPATH / 'rectified_imgs').glob('*.png'))]
    B_filenames = [item.name for item in list((B_DPATH / 'rectified_imgs').glob('*.png'))]

    # Compress images and save
    for a_fname, b_fname in zip(A_filenames, B_filenames):
        opath_img_a = A_DPATH / 'compressed_rectified_imgs' / a_fname
        opath_img_b = B_DPATH / 'compressed_rectified_imgs' / b_fname

        # Create directory if needed
        if not opath_img_a.parent.exists():
            opath_img_a.parent.mkdir(exist_ok=True, parents=True)
        if not opath_img_b.parent.exists():
            opath_img_b.parent.mkdir(exist_ok=True, parents=True)

        if opath_img_b.exists() and opath_img_a.exists():
            continue

        print('compressing rectified images for:', a_fname)

        img_a = plt.imread(A_DPATH / 'rectified_imgs' / a_fname)
        img_b = plt.imread(B_DPATH / 'rectified_imgs' / b_fname)

        # Compress images
        compress_img_a = cv2.resize(img_a, (504, 504))
        compress_img_b = cv2.resize(img_b, (504, 504))

        plt.imsave(opath_img_a, compress_img_a)
        plt.imsave(opath_img_b, compress_img_b)


if __name__ == '__main__':
    orientate_cameras()
    compress_imgs()
