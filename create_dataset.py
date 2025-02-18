import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from fisheye_to_perspective_utils import extract_cube_map
import yaml

#Get paths
A_DPATH = Path('/gws/nopw/j04/iecdt/computer-vision-data/cam_a/imgs')
B_DPATH = Path('/gws/nopw/j04/iecdt/computer-vision-data/cam_b/imgs')

#Find common filenames
A_filenames = []
B_filenames = []

for item in list(A_DPATH.glob('*.png')):
    A_filenames.append(item.parts[-1])

for item in list(B_DPATH.glob('*.png')):
    B_filenames.append(item.parts[-1])

common = [item for item in A_filenames if item in B_filenames]

#Extract camera parameters
with open("/gws/nopw/j04/iecdt/computer-vision-data/fisheye_calib_A.yml", 'r') as f:
    calib_A = yaml.safe_load(f)

with open("/gws/nopw/j04/iecdt/computer-vision-data/fisheye_calib_B.yml", 'r') as f:
    calib_B = yaml.safe_load(f)

#Get rotations
transform = np.loadtxt('/gws/nopw/j04/iecdt/computer-vision-data/T_rel.txt')

#Process images and save
for i in range(262, len(common)):
    img_a = extract_cube_map(plt.imread(A_DPATH / common[i]), calib_A, rotate=np.rad2deg(np.arctan(transform[0,-1]/transform[1,-1])))
    img_b = extract_cube_map(plt.imread(B_DPATH / common[i]), calib_B, rotate=np.rad2deg(np.arccos(transform[0,0]))+np.rad2deg(np.arctan(transform[0,-1]/transform[1,-1])))
    plt.imsave(Path('dataset/cam_a/' + common[i]), img_a[-1])
    plt.imsave(Path('dataset/cam_b/' + common[i]), img_b[-1])
    print(i)
