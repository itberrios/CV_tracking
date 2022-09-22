'''
This script contains various utility functions for working with the KITTI dataset
'''
import os
import cv2
import numpy as np
import torch
from sklearn import linear_model
# import pymap3d as pm


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
# time functions
get_total_seconds = lambda hms: hms[0]*60*60 + hms[1]*60 + hms[2]


def timestamps2seconds(timestamp_path):
    ''' Reads in timestamp path and returns total seconds (does not account for day rollover '''
    timestamps = pd.read_csv(timestamp_path, 
                             header=None, 
                             squeeze=True).astype(object) \
                                          .apply(lambda x: x.split(' ')[1]) 
    
    # Get Hours, Minutes, and Seconds
    hours = ser.apply(lambda x: x.split(':')[0]).astype(np.float64)
    minutes = ser.apply(lambda x: x.split(':')[1]).astype(np.float64)
    seconds = ser.apply(lambda x: x.split(':')[2]).astype(np.float64)

    hms_vals = np.vstack((hours, minutes, seconds)).T
    
    total_seconds = np.array(list(map(get_total_seconds, hms_vals)))
    
    return total_seconds

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
    camera =  T @ xyz

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


def transform_uvz(uvz, T):
    ''' Transforms the uvz coordinates to xyz coordinates. The xyz coordinate
        frame is specified by the transformation matrix T. The transformation
        may include:
            uvz -> xyz (LiDAR)
            uvz -> xyz (IMU)

        Inputs: uvz (Nx3) array of uvz coordinates
        Outputs: xyz (Nx3) array of xyz coordinates
        '''
    # covnert to homogeneous representation
    uvzw = np.hstack((uvz[:, :2] * uvz[:, 2][:, None], 
                      uvz[:, 2][:, None],
                      np.ones((len(uvz[:, :3]), 1))))
    
    # perform homogeneous transformation
    xyzw = T @ uvzw.T
    
    # get xyz coordinates
    xyz = xyzw[:3, :].T

    return xyz



# ============================================================================================
# plotting functions (place these in KITTI plot utils
from matplotlib import cm

# get color map function
rainbow_r = cm.get_cmap('rainbow_r', lut=100)
get_color = lambda z : [255*val for val in rainbow_r(int(z.round()))[:3]]

def draw_velo_on_image(velo_uvz, image, color_map=get_color):
   
    # unpack LiDAR points
    u, v, z = velo_uvz

    # draw LiDAR point cloud on blank image
    for i in range(len(u)):
        cv2.circle(image, (int(u[i]), int(v[i])), 1, 
                   color_map(z[i]), -1);

    return image


# ============================================================================================
# pipeline functions place these in KITTI LIDAR utils

def project_velobin2uvz(bin_path, T_uvz_velo, image, remove_plane=True):
    ''' Projects LiDAR point cloud onto the image coordinate frame (u, v, z)
        '''

    # get homogeneous LiDAR points from bin file
    xyzw = bin2xyzw(bin_path, remove_plane)

    # project velo (x, z, y, w) onto camera (u, v, z) coordinates
    velo_uvz = xyzw2camera(xyzw, T_uvz_velo, image, remove_outliers=True)
    
    return velo_uvz


def get_velo_xyz(image, velo_uvz, bboxes, draw=True):
    ''' Obtains LiDAR xyz measurements for each detected object in the image,
        also gets the LiDAR xyz projected onto the camera reference frame as uvz 
        Inputs:
          image - input image for detection 
          velo_uvz - LiDAR coordinates projected to camera reference
          bboxes - xyxy bounding boxes form detections from yolov5 model output
          draw - (_Bool) draw measured depths on image
        Outputs:
          image - input image with distances drawn at the center of each 
                  bounding box
          bboxes_out - bboxes with velo (u,v,z) and (x,y,z) object centers
                  added to the array
        '''

    # unpack LiDAR camera coordinates
    u, v, z = velo_uvz

    # get new output
    bboxes_out = np.zeros((bboxes.shape[0], bboxes.shape[1] + 6))
    bboxes_out[:, :bboxes.shape[1]] = bboxes

    # iterate through all detected bounding boxes
    for i, bbox in enumerate(bboxes):
        pt1 = torch.round(bbox[0:2]).to(torch.int).numpy()
        pt2 = torch.round(bbox[2:4]).to(torch.int).numpy()

        # get center location of the object on the image
        obj_x_center = (pt1[1] + pt2[1]) / 2
        obj_y_center = (pt1[0] + pt2[0]) / 2

        # now get the closest LiDAR points to the center
        center_delta = np.abs(np.array((v, u)) 
                              - np.array([[obj_x_center, obj_y_center]]).T)
        
        # choose coordinate pair with the smallest L2 norm
        min_loc = np.argmin(np.linalg.norm(center_delta, axis=0))

        # get LiDAR location in image/camera space
        velo_depth = z[min_loc]; # LiDAR depth in camera space
        velo_uvz_location = np.array([v[min_loc], u[min_loc], velo_depth])

        # convert uvz location to LiDAR xyz location
        velo_uvzw_location = np.hstack((velo_uvz_location[:2] * velo_uvz_location[2],
                                        velo_uvz_location[2],
                                        1))[:, None]
        velo_xyzw = T_uvz_velo_inv @ velo_uvzw_location

        # add velo (u, v, z) to bboxes 
        bboxes_out[i, -6:-3] = velo_uvz_location

        # add velo (x, y, z) to bboxes
        bboxes_out[i, -3:] = velo_xyzw[:3].squeeze()

        # draw depth on image at center of each bounding box
        # This is depth as perceived by the camera
        if draw:
            object_center = (np.round(obj_y_center).astype(int), 
                             np.round(obj_x_center).astype(int))
            cv2.putText(image, 
                        '{0:.2f} m'.format(velo_depth), 
                        object_center,
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 0, 0), 2, cv2.LINE_AA)    
            
    return image, bboxes_out


# ============================================================================================
# ============================================================================================
# load coordinate transformations


# camera transformations
with open('2011_10_03/calib_cam_to_cam.txt','r') as f:
    calib = f.readlines()

# get projection matrices (rectified left camera --> left camera (u,v,z))
P_rect2_cam2 = np.array([float(x) for x in calib[25].strip().split(' ')[1:]]).reshape((3,4))


# get rectified rotation matrices (left camera --> rectified left camera)
R_ref0_rect2 = np.array([float(x) for x in calib[24].strip().split(' ')[1:]]).reshape((3, 3,))

# add (0,0,0) translation and convert to homogeneous coordinates
R_ref0_rect2 = np.insert(R_ref0_rect2, 3, values=[0,0,0], axis=0)
R_ref0_rect2 = np.insert(R_ref0_rect2, 3, values=[0,0,0,1], axis=1)


# get rigid transformation from Camera 0 (ref) to Camera 2
R_2 = np.array([float(x) for x in calib[21].strip().split(' ')[1:]]).reshape((3,3))
t_2 = np.array([float(x) for x in calib[22].strip().split(' ')[1:]]).reshape((3,1))

# get cam0 to cam2 rigid body transformation in homogeneous coordinates
T_ref0_ref2 = np.insert(np.hstack((R_2, t_2)), 3, values=[0,0,0,1], axis=0)

# LiDAR and IMU rgid body transformations
T_velo_ref0 = get_rigid_transformation(r'2011_10_03/calib_velo_to_cam.txt')
T_imu_velo = get_rigid_transformation(r'2011_10_03/calib_imu_to_velo.txt')


# LiDAR <--> Camera 2 transformations
# transform from velo (LiDAR) to left color camera (shape 3x4)
T_velo_cam2 = P_rect2_cam2 @ R_ref0_rect2 @ T_ref0_ref2 @ T_velo_ref0 

# homogeneous transform from left color camera to velo (LiDAR) (shape: 4x4)
T_cam2_velo = np.linalg.inv(np.insert(T_velo_cam2, 3, values=[0,0,0,1], axis=0)) 


# IMU <--> Camera 2 transformations
# transform from IMU to left color camera (shape 3x4)
T_imu_cam2 = T_velo_cam2 @ T_imu_velo

# homogeneous transform from left color camera to IMU (shape: 4x4)
T_cam2_imu = np.linalg.inv(np.insert(T_imu_cam2, 3, values=[0,0,0,1], axis=0)) 

# ============================================================================================


