import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


# helper functions
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
    """ Computes Cost between 2 bounding boxes
        """
    iou_cost = compute_iou(box1, box2)
    if (iou_cost >= iou_thresh):
      return iou_cost
    else:
      return 0


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
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
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
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
    Function goal: Define a Hungarian Matrix with IOU as a metric and return, for each box, an id
    Outputs:
      matches - matched track indexes (old , new)
      unmatched_detections - unmatched detection indexes
      unmatched_tracks - unmatched track indexes
    """
    if (len(new_boxes) == 0) and (len(old_boxes) == 0):
        return [], [], []
    elif(len(old_boxes)==0):
        return [], np.arange(0, len(new_boxes)), []
    elif(len(new_boxes)==0):
        return [], [], np.arange(0, len(old_boxes))

    # Define a new cost Matrix nxm with old and new boxes
    cost_matrix = np.zeros((len(old_boxes),len(new_boxes)),dtype=np.float32)

    # Go through boxes and store the IOU value for each box 
    # You can also use the more challenging cost but still use IOU as a reference for convenience (use as a filter only)
    for i,old_box in enumerate(old_boxes):
        for j,new_box in enumerate(new_boxes):
            cost_matrix[i][j] = compute_cost(old_box, new_box)
    
    # Find optimal assignments with the  Hungarian Algorithm
    hungarian_row, hungarian_col = linear_sum_assignment(-cost_matrix)
    hungarian_matrix = np.array(list(zip(hungarian_row, hungarian_col)))

    # Create new unmatched lists for old and new boxes
    matches, unmatched_detections, unmatched_tracks = [], [], []

    # Go through the Hungarian Matrix, if matched element has IOU < threshold (0.3), add it to the unmatched 
    # Else: add the match    
    for h in hungarian_matrix:
        if(cost_matrix[h[0],h[1]]<thresh):
            # unmatched_tracks.append(old_boxes[h[0]])
            # unmatched_detections.append(new_boxes[h[1]])
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
          # unmatched_tracks.append(trk)
          unmatched_tracks.append(t)
    
    # Go through new boxes, if no matched tracking, add it to the unmatched_new_boxes
    for d, det in enumerate(new_boxes):
        if(d not in hungarian_matrix[:,1]):
            # unmatched_detections.append(det)
            unmatched_detections.append(d)
    
    return matches, unmatched_detections, unmatched_tracks


class Obstacle():
    count = 0
    def __init__(self, box, cat):
        ''' 
            box - bounding box ccordinates [x1, y1, x2, y2]
            cat - (int) MSCOCO category
            
            potential adds
            age - track age, number of frames track has been observed
            unmatched_age - number of frames track has not been observed
            fov - (_Bool) flag to denote whether the object is approaching the edge of the FOV
            '''
        self.box = box
        self.cat = cat
        self.age = 0
        self.unmatched_age = 0
        self.fov = 0

        self.id = Obstacle.count
        Obstacle.count += 1

        self.time_since_update = 0
        self.history = []
        self.hits = 1
        self.hit_streak = 0

        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4) 
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        # get initial state
        self.kf.x[:4] = convert_bbox_to_z(box)


    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)



class Sort():
    def __init__(self, max_age=1, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.frame_count = 0

        # reset obstacle count
        Obstacle.count = 0

    def update(self, detections=np.empty((0, 5))):
        """ Performs track update
            Inputs:
              detections - array of detections in the form of: [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
            
            """
        self.frame_count += 1

        # get predicted locations from existing trackers.
        # trks = np.zeros((len(self.tracks), 5))
        # to_del = []
        # ret = []
        # for t, trk in enumerate(trks):
        #     pos = self.trackers[t].predict()[0]
        #     trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
        #     if np.any(np.isnan(pos)):
        #         to_del.append(t)
        # trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        # for t in reversed(to_del):
        #     self.trackers.pop(t)

        # update track locations
        for t, trk in enumerate(self.tracks):
            trk.predict()


        # get bounding boxes for currently tracked objects
        old_bboxes = np.array([trk.box for trk in self.tracks])

        # associate boxes to known tracks
        matches, unmatched_detections, unmatched_tracks = \
            associate(old_bboxes, detections[:, :4], thresh=self.iou_threshold)
        
        # update matches
        for m in matches:
            # update box positions
            self.tracks[m[0]].update(detections[m[1], :4])

        # initialize new tracks from unmatched detections
        for d_idx in unmatched_detections:
            box = detections[d_idx][:4]
            cat = detections[d_idx][-1]
            self.tracks.append(Obstacle(box=box, cat=cat))

        # handle unmatched tracks (handle this inside Obstacle)
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].time_since_update += 1
            self.tracks[t_idx].hit_streak = 0

        # manage tracks
        current_tracks = []
        i = len(self.tracks)
        for trk in reversed(self.tracks):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                current_tracks.append(np.concatenate((d,[trk.id+1, trk.cat])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.tracks.pop(i)
        if(len(current_tracks)>0):
            return np.concatenate(current_tracks)
        return np.empty((0,5))
