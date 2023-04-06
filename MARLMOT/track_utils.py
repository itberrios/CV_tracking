"""
    Utility functions for MARLMOT
"""


import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


# helper functions
sigmoid = lambda x: 1/(1 + np.exp(-x))


def compute_iou(box1, box2):
    """ Obtains Intersection over union (IOU) of 2 bounding boxes
        Inputs are in the form of:
            xmin, ymin, xmax, ymax = box
        """
    x11, y11, x21, y21 = box1
    x12, y12, x22, y22 = box2

    # get box points of intersection
    xi1 = max(x11, x12) # top left
    yi1 = max(y11, y12)
    xi2 = min(x21, x22) # bottom right
    yi2 = min(y21, y22)

    # compute intersectional area
    inter_area = max((xi2 - xi1 + 1), 0) * max((yi2 - yi1 + 1), 0)
    if inter_area == 0:
        return inter_area

    # compute box areas
    box1_area = (x21 - x11 + 1) * (y21 - y11 + 1)
    box2_area = (x22 - x12 + 1) * (y22 - y12 + 1)

    # return iou
    return inter_area / (box1_area + box2_area - inter_area)


def compute_cost(box1, box2, iou_thresh=0.3):
    """ Computes Cost between 2 bounding boxes  """
    iou_cost = compute_iou(box1, box2)
    if (iou_cost >= iou_thresh):
      return iou_cost
    else:
      return 0


def convert_bbox_to_z(bbox):
    """ Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
        """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
    """ Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
        """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


def associate(old_boxes, new_boxes, thresh=0.3):
    """ Associates old boxes with new boxes
        Inputs:
            old_boxes - old bounding boxes at time t - 1
            new_boxes - new bounding boxes at time t
            thresh - min threshold needed for bbox association
        Function goal: Define a Hungarian Matrix with IOU as a metric and return, for each box, an id
        Outputs:
            matches - matched track indexes (old , new)
            unmatched_detections - unmatched detection indexes
            unmatched_tracks - unmatched track indexes
            cost_matrix - cost of each association indexes by the amtch indexes
        """
    if (len(new_boxes) == 0) and (len(old_boxes) == 0):
        return [], [], [], []
    elif(len(old_boxes)==0):
        return [], np.arange(0, len(new_boxes)), [], []
    elif(len(new_boxes)==0):
        return [], [], np.arange(0, len(old_boxes)), []

    # Define a new cost Matrix nxm with old and new boxes
    cost_matrix = np.zeros((len(old_boxes),len(new_boxes)),dtype=np.float32)

    # Go through boxes and store the IOU value for each box 
    # You can also use the more challenging cost but still use IOU as a reference for convenience (use as a filter only)
    for i,old_box in enumerate(old_boxes):
        for j,new_box in enumerate(new_boxes):
            cost_matrix[i][j] = compute_cost(old_box, new_box, iou_thresh=thresh)
    
    # Find optimal assignments with the  Hungarian Algorithm
    hungarian_row, hungarian_col = linear_sum_assignment(-cost_matrix)
    hungarian_matrix = np.array(list(zip(hungarian_row, hungarian_col)))

    # Create new unmatched lists for old and new boxes
    matches, unmatched_detections, unmatched_tracks = [], [], []

    # Go through the Hungarian Matrix, if matched element has IOU < threshold (0.3), add it to the unmatched 
    # Else: add the match    
    for h in hungarian_matrix:
        if(cost_matrix[h[0],h[1]] < thresh):
            unmatched_tracks.append(h[0])
            unmatched_detections.append(h[1])
        else:
            matches.append(h.reshape(1,2))
    
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    
    # Go through old boxes, if no matched detection, add it to the unmatched_old_boxes
    for t, trk in enumerate(old_boxes):
        if(t not in hungarian_matrix[:,0]):
          unmatched_tracks.append(t)
    
    # Go through new boxes, if no matched tracking, add it to the unmatched_new_boxes
    for d, det in enumerate(new_boxes):
        if(d not in hungarian_matrix[:,1]):
            unmatched_detections.append(d)
    
    return matches, unmatched_detections, unmatched_tracks, cost_matrix

# obstacle class
class Obstacle():
    count = 0
    def __init__(self, box, cat):
        """ Stores information for a single obstacle/track
            Args:
                box - (array) bounding box coordinates [x1, y1, x2, y2]
                cat - (int) MSCOCO category
            
            Attributes:

                age - (int) total number of frames that the obstacle has been alive
                id - (int) Unique Track ID 
                hits - (int) number of detections that have been associated to the track
                hit_streak - (int) number of consecutive hits
                detection - (array) most recent associated detection vector [x, y, area, aspect ratio, cost]
                            This will be updated externally (usually by a Tracker)
                history - (list) contains all obstacle bbox locations
                time_since_last_update - (int) number of frames since the most recent associated detection
                track_mode - (int) status of current track (managed externally)
                            0 - inactive, 1 - visible, 2 - hidden
                observation - (array) 18x1 observation vector (for RL Agent)
            """
        self.cat = cat
        self.age = 0 # age of -1 indicates track to be deleted

        self.id = Obstacle.count
        Obstacle.count += 1

        self.hits = 1 
        self.hit_streak = 0 # final position in observation vector

        # most recent detection ([x, y, area, aspect ratio, conf, cost])
        self.detection = None 

        self.time_since_update = 0 
        self.history = []
        
        # track mode always starts as 0 and can be 0,1,2
        # maybe assume all new tracks are active since they had to be detected to begin with
        self.track_mode = 1 # 0 
        
        # current track observation
        self.observation = np.zeros((18, 1))

        # get initial constant velocity model
        self.reset_kf(box)


    def reset_kf(self, box):
        """ Reinitializes the Kalman Filter to the location specified by
            the input bounding box.
            Useful when the motion model fails.
            Inputs: box - (bounding box coordinates (x1, y1, x2, y2))
            """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4) 
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        # get initial state
        self.kf.x[:4] = convert_bbox_to_z(box)

    def update(self, box):
        """ Updates the Kalman Filter
            """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(box))

    def predict(self):
        """ Advances the state vector and returns the predicted 
            bounding box estimate.
            """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """ Returns the current bounding box estimate. """
        return convert_x_to_bbox(self.kf.x)
    
    def update_observation(self):
        """ Obtains current observation for track management """
        self.observation[0:7] = self.kf.x 
        self.observation[7:13] = self.detection

        # update 1 hot encoded track mode
        if self.track_mode == 0:
            self.observation[13] = 1
            self.observation[14:16] = 0
        elif self.track_mode == 1:
            self.observation[14] = 1
            self.observation[[13,15]] = 0
        elif self.track_mode == 2:
            self.observation[15] = 1
            self.observation[13:15] = 0

        # number of timesteps since last association to agent while active or hidden
        self.observation[16] = self.time_since_update

        # number of consequtive timesteps that an association is received while active or hidden
        self.observation[17] = self.hit_streak



class HungarianTracker():
    def __init__(self, iou_threshold=0.3, min_age=1):
        """ Tracks obstacle objects with Hungarian association to 
            args:
                iou_threshold - min threshold needed to perform IOU association
                min_age - minium age for a track to be valid (0 indexed)
                    Protects against sporadic detections. setting the min age
                    helps our model learn, by avoiding noisy sporadic detections
            """
        self.iou_threshold = iou_threshold
        self.min_age = min_age
        self.tracks = []
        self.frame_count = 0

        # reset obstacle count to ensure IDs start from 0
        Obstacle.count = 0

    def reset(self):
        """ resets all tracks and frame count """
        self.tracks = []
        self.frame_count = 0

        # reset obstacle count to ensure IDs start from 0
        Obstacle.count = 0


    def update(self, detections=np.empty((0, 6))):
        """ Performs track update
            Inputs:
              detections - array of detections in the form of: [[x1,y1,x2,y2,cat,score],[x1,y1,x2,y2,cat,score],...]
            Outputs: A list of track objects to be managed by the RL tracker
            """
        self.frame_count += 1

        # update track locations and get bounding boxes
        old_bboxes = []
        for track in self.tracks:
            # remove tracks that have been marked for deletion (age = -1)
            if track.age >= 0:
                old_bboxes.append(track.predict()[0]) # always make a prediction
                
        # associate boxes to known tracks
        matches, \
        unmatched_detections, \
        unmatched_tracks, \
        cost_matrix = associate(old_bboxes, 
                                np.round(detections[:, :4]).astype(int), 
                                thresh=self.iou_threshold)

        # get associated tracks
        tracks = []
        for m in matches:
            track = self.tracks[m[0]]
            cost = cost_matrix[m[0], m[1]]
            bbox = detections[m[1], :4]
            track.detection = np.vstack((convert_bbox_to_z(bbox), 
                                         detections[m[1], 5], # confidence score
                                         cost))
            tracks.append(self.tracks[m[0]])

        # get new tracks 
        new_tracks = []
        for d in unmatched_detections:
            box = np.round(detections[d, :4]).astype(int)
            cat = detections[d, 4]
            cost = 0
            track = Obstacle(box=box, cat=cat)

            track.detection = np.vstack((convert_bbox_to_z(detections[d, :4]), 
                                         detections[d, 5], # confidence score
                                         cost))
            new_tracks.append(track)

        # get unmatched tracks
        unmatched_tracks = []
        for t in unmatched_tracks:
            track = self.tracks[t]
            track.detection = 0
            unmatched_tracks.append(track)

        # update tracks list with all tracks
        self.tracks = tracks + new_tracks + unmatched_tracks

        # get valid tracks
        current_tracks = []
        for track in self.tracks:
            if track.age >= self.min_age:
                track.update_observation() # update observation vector
                current_tracks.append(track)

        # return tracks, new_tracks, unmatched_tracks
        return current_tracks

        