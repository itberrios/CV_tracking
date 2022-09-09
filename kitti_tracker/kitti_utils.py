'''
This script contains various utility functions for working with the KITTI dataset
'''
import os
import cv2
import numpy as np
from sklearn import linear_model


def test_func(a, b):
    ''' Trivial function to test imports in Google Colab '''
    return a + b

# ============================================================================================
# GPS/IMU functions

def get_oxts(oxt_path):
    ''' Obtains the oxt info from a single oxt path '''
    with open(oxt_path) as f:
        oxts = f.readlines()
        
    oxts = oxts[0].strip().split(' ')
    oxts = np.array(oxts).astype(float)
    
    return oxts

# ============================================================================================
# file access functions

def bin2xyzw(bin_path, remove_plane=False):
    ''' Reads LiDAR bin file and returns homogeneous (x,y,z,1) LiDAR points'''
    # read in LiDAR data
    scan_data = np.fromfile(bin_path, dtype=np.float32).reshape((-1,4))

    # get x,y,z LiDAR points (x, y, z) --> (front, left, up)
    xyz = scan_data[:, 0:3] 

    # delete negative liDAR points
    xyz = np.delete(xyz, np.where(xyz[3, :] < 0), axis=1)

    # use ransac to remove ground plane
    if remove_plane:
        ransac = linear_model.RANSACRegressor(
                                      linear_model.LinearRegression(),
                                      residual_threshold=0.1,
                                      max_trials=5000
                                      )

        X = xyz[:, :2]
        y = xyz[:, -1]
        ransac.fit(X, y)
        
        # remove outlier points (i.e. remove ground plane)
        mask = ransac.inlier_mask_
        xyz = xyz[~mask]

    # conver to homogeneous LiDAR points
    xyzw = np.insert(xyz, 3, 1, axis=1).T 

    return xyzw



# ============================================================================================
# calibration functions

def decompose_projection_matrix(P):    
    K, R, T, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    T = T/T[3]

    return K, R, T


def get_rigid_transformation(calib_path):
    ''' Obtains rigid transformation matrix in homogeneous coordinates (combination of
        rotation and translation.
        Used to obtain:
            - LiDAR to camera reference transformation matrix 
            - IMU to LiDAR reference transformation matrix
        '''
    with open(calib_path, 'r') as f:
        calib = f.readlines()

    R = np.array([float(x) for x in calib[1].strip().split(' ')[1:]]).reshape((3, 3))
    t = np.array([float(x) for x in calib[2].strip().split(' ')[1:]])[:, None]

    T = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))
    
    return T

# ============================================================================================
# coordiante transformations

def xyzw2camera(xyz, T, image=None, remove_outliers=True):
    ''' maps xyxw homogeneous points to camera (u,v,z) space. The xyz points can 
        either be velo/LiDAR or GPS/IMU, the difference will be marked by the 
        transformation matrix T.
        '''
    # convert to (left) camera coordinates
    camera =  T_mat @ xyz

    # delete negative camera points
    camera  = np.delete(camera , np.where(camera [2,:] < 0)[0], axis=1) 

    # get camera coordinates u,v,z
    camera[:2] /= camera[2, :]

    # remove outliers (points outside of the image frame)
    if remove_outliers:
        u, v, z = camera
        img_h, img_w, _ = image.shape
        u_out = np.logical_or(u < 0, u > img_w)
        v_out = np.logical_or(v < 0, v > img_h)
        outlier = np.logical_or(u_out, v_out)
        camera = np.delete(camera, np.where(outlier), axis=1)

    return camera

# ============================================================================================
# pipeline functions

def project_velobin2uvz(bin_path, T_uv_velo, image, remove_plane=True):
    ''' Projects LiDAR point cloud onto the image coordinate frame (u, v, z)
        '''

    # get homogeneous LiDAR points from bin file
    xyzw = bin2xyzw(bin_path, remove_plane)

    # project velo (x, z, y, w) onto camera (u, v, z) coordinates
    velo_uvz = xyzw2camera(xyzw, T_uv_velo, image, remove_outliers=True)
    
    return velo_uvz


def get_distances(image, velo_uvz, bboxes, draw=True):
    ''' Obtains distance measurements for each detected object in the image 
        Inputs:
          image - input image for detection 
          velo_uvz - LiDAR coordinates projected to camera reference
          bboxes - xyxy bounding boxes form detections from yolov5 model output
          draw - (_Bool) draw measured depths on image
        Outputs:
          image - input image with distances drawn at the center of each 
                  bounding box
        '''

    # unpack LiDAR camera coordinates
    u, v, z = velo_uvz

    # get new output
    bboxes_out = np.zeros((bboxes.shape[0], bboxes.shape[1] + 3))
    bboxes_out[:, :bboxes.shape[1]] = bboxes

    # iterate through all detected bounding boxes
    for i, bbox in enumerate(bboxes):
        pt1 = torch.round(bbox[0:2]).to(torch.int).numpy()
        pt2 = torch.round(bbox[2:4]).to(torch.int).numpy()

        # get center location of the object on the image
        x_center = (pt1[1] + pt2[1]) / 2
        y_center = (pt1[0] + pt2[0]) / 2

        # now get the closest LiDAR points to the center
        center_delta = np.abs(np.array((v, u)) 
                              - np.array([[x_center, y_center]]).T)
        
        # choose coordinate pair with the smallest L2 norm
        min_loc = np.argmin(np.linalg.norm(center_delta, axis=0))

        # get LiDAR location in image/camera space
        velo_depth = z[min_loc]; # LiDAR depth in camera space
        velo_location = np.array([v[min_loc], u[min_loc], velo_depth])

        # add velo (u, v, z) to bboxes
        bboxes_out[i, -3:] = velo_location

        # draw depth on image at center of each bounding box
        if draw:
            object_center = (np.round(y_center).astype(int), 
                             np.round(x_center).astype(int))
            cv2.putText(image, 
                        '{0:.2f} m'.format(velo_depth), 
                        object_center,
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 0, 0), 2, cv2.LINE_AA)    
            
    return image, bboxes_out


