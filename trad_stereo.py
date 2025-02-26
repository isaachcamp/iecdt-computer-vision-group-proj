from pathlib import Path

import numpy as np
import pandas as pd
import cv2

from sklearn.metrics import accuracy_score, confusion_matrix


IMG_PATH = Path('/gws/nopw/j04/iecdt/computer-vision-data')
LABELS_PATH = Path('/gws/nopw/j04/iecdt/JERMIT_the_frog/')


def preprocess_img(img):
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_imgs(fname):
    img_a = cv2.imread(IMG_PATH / f'cam_a/compressed_rectified_imgs/{fname}.png')
    img_b = cv2.imread(IMG_PATH / f'cam_b/compressed_rectified_imgs/{fname}.png')
    return preprocess_img(img_a), preprocess_img(img_b)

def main():

    # Set camera parameters (from calibration)
    img_px_width = 3040
    fov = 90
    focal_length = img_px_width / (2 * np.tan(np.deg2rad(fov / 2)))
    baseline = 271  # Distance between cameras in meters

    # StereoSGBM parameters
    num_disparities = 16*4
    blocksize = 5

    # Load labels
    labels = pd.read_csv(
        LABELS_PATH / 'hydrometeors_time_aligned_classes.csv',
        index_col=0
    )
    dset_size = labels.shape[1]
    height_bins = np.insert(labels.index.values, 0, 0)

    sgm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,  # Should be a multiple of 16
        blockSize=blocksize,
        P1=8 * 3 * blocksize**2,  # 8 * number_of_channels * blockSize^2
        P2=32 * 3 * blocksize**2,
        disp12MaxDiff=7,
        uniquenessRatio=7,
        speckleWindowSize=50,
        speckleRange=3
    )

    accuracies = np.zeros(dset_size)
    agg_confusion_matrix = np.zeros((2, 2, dset_size))
    y_pred = np.zeros_like(labels.values)

    for i in range(dset_size):
        img_a, img_b = get_imgs(labels.columns[i])
        y_true = labels.iloc[:, i].values

        # Calculate disparity map
        disparity = sgm.compute(img_a, img_b).astype(np.float32)

        disparity[disparity <= 0] = np.nan # Avoid division by zero

        # Calculate depth map
        depth_map = (focal_length * baseline) / disparity

        # Calculate binary prediction for presence of hydrometeors
        counts, _ = np.histogram(depth_map.flatten(), bins=height_bins)
        y_pred[:, i] = (counts >= 1).astype(int)

        # Evalaute for single image
        accuracies[i] = accuracy_score(y_true, y_pred[:,i])
        agg_confusion_matrix[:, :, i] = confusion_matrix(y_true, y_pred[:,i], labels=[0, 1])

        if i % 100 == 0:
            print(f'Processed {i} images')

    np.save(LABELS_PATH / 'stereo_preds.npy', y_pred)
    np.save(LABELS_PATH / 'stereo_accuracies.npy', accuracies)
    np.save(LABELS_PATH / 'stereo_confusion_matrix.npy', agg_confusion_matrix)


if __name__ == '__main__':
    main()
