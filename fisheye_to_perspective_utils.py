from omnicalib.undistort import Projection, Undistort
from omnicalib.undistort import get_view_vectors
from numpy.polynomial.polynomial import polyval
import numpy as np
import cv2
import imageio
import yaml
import torch
import os
import argparse

def polar_to_cartesian(magnitude, angle_deg):
    """
    Convert a polar coordinate (magnitude, angle in degrees) to cartesian (x, y).
    """
    angle_rad = np.deg2rad(angle_deg)
    x = magnitude * np.cos(angle_rad)
    y = magnitude * np.sin(angle_rad)
    return (x, y)

def extract_cube_map(img, calibration, invert=False, size=(3040, 3040), fov=90, rotate=5):
    img = cv2.resize(img, size)
    
    poly_rz = torch.tensor(calibration['poly_radius_to_z'], dtype=torch.float32)
    poly_thetar = poly_rz.new_tensor(calibration['poly_incident_angle_to_radius'])
    principal_point = poly_rz.new_tensor(calibration['principal_point'])
    view_shape = size
    fovx = torch.deg2rad(poly_rz.new_tensor(fov // 2))
    
    view_angles = [-180, -90, 0, 90]
    view_angles = [x + rotate for x in view_angles]
    side_views = [(polar_to_cartesian(1, x)[0], -polar_to_cartesian(1, x)[1], 0) for x in view_angles]
    
    side_views = poly_rz.new_tensor(side_views)

    if invert:
        side_views *= -1

    # Add an extra view for the sky (using the same angle as the second side view)
    views = torch.cat((side_views, side_views[1:2, :]), dim=0)
    
    projection = Projection(poly_thetar, poly_rz, principal_point)
    
    undistort = Undistort(
        projection,
        view_shape,
        fovx,
        views
    )

    centre_x = principal_point[0]
    centre_y = principal_point[1]

    offset_angle = 90
    offset_magnitude = polyval(offset_angle / 180 * np.pi, poly_thetar)
    offsets = [polar_to_cartesian(offset_magnitude, x) for x in view_angles]

    # Negate y coordinate to match OpenCV image coordinate convention
    p_image = poly_rz.new_tensor((
        (centre_x + offsets[0][0], centre_y - offsets[0][1]),
        (centre_x + offsets[1][0], centre_y - offsets[1][1]),
        (centre_x + offsets[2][0], centre_y - offsets[2][1]),
        (centre_x + offsets[3][0], centre_y - offsets[3][1]),
        (centre_x, centre_y)
    ))
    print(p_image)
    # Prepare the input image for PyTorch and run the undistortion
    M, undistorted = undistort(
        torch.from_numpy(img).permute(2, 0, 1)[None].to(poly_rz),
        p_image
    )
    
    # Convert the result from tensor back to numpy arrays (list of faces)
    undistorted = list(undistorted.permute(0, 2, 3, 1).cpu().numpy())
    
    return undistorted
