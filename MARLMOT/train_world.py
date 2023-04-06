""" 
    Environment for training.

    Uses MOT15 detections

"""

import numpy as np
import pandas as pd
from track_utils import *


class TrainWorld():

    def __init__(self, tracker, ground_truth, detections, frame_size):
        """ Training World for visual Tracking. The ground_truth and detections
            correspond to a single video of tracks and detections.
            Args:
                tracker - Tracker Object
                ground_truth - DataFrame of Ground Truth tracks at every frame
                detections - DataFrame of detections at every frame
                frame_size - frame size list (num rows, num cols)

            Attributes
                frame - current frame index
                current_tracks - confirmed track dictions (id -> track object)
                truth_tracks - current truth tracks
                truth_bboxes - current truth bboxes
                id_map - dict that maps tracker IDs to ground truth IDs
                mismatch_errors - number of current mismatch errors
                matches - number of matched tracks
                false_positives - number of false positives
                missed_tracks - number of missed tracks
                cost_penalty - cost penalty (1 - cost) --> lower is better
            """
        self.tracker = tracker   # tracker class
        self.ground_truth = ground_truth # ground truth DataFrame
        self.detections = detections # DataFrame of detections for offline training
        self.frame_size = frame_size # frame size (num_rows, num_cols)

        self.frame = 0 # current frame index
        self.current_tracks = {} # confirmed tracks
        self.truth_tracks = [] # current truth tracks
        self.truth_bboxes = [] # current truth bboxes
        self.id_map = {} # maps tracker IDs to ground truth IDs
        self.mismatch_errors = 0 # number of current mismatch errors
        self.matches = []
        self.false_positives = []
        self.missed_tracks = []
        self.cost_penalty = 0
        self.total_age_diff = 0 # total age different between current and GT tracks

        # ensure that the Tracker is reset
        self.tracker.reset()


    def _update_gt_bboxes(self):
        """ Update Ground Truth bboxes """
        gt_bbox = []
        # draw ground truth on frame
        for id in self.truth_tracks.id:
            track = self.truth_tracks[self.truth_tracks.id == id]
            pt1 = track.iloc[0, 2:4].to_numpy().astype(int)
            pt2 = pt1 + track.iloc[0, 4:6].to_numpy().round().astype(int)
            gt_bbox.append([pt1[0], pt1[1], pt2[0], pt2[1]])

        # convert to array
        self.truth_bboxes = np.array(gt_bbox)
       
    
    def _get_detection_bboxes(self, detections):
        """ Obtains Detection bboxes and confidence 
            Inputs:
                detections - DataFrame of detections at current frame
            Outputs:
                detections - (Nx5) array of detections in the form of:
                    (x1, y1, x2, y2, category, confidence)
                    (left, top, right, bottom, category, confidence)
            """
        pt1 = detections.iloc[:, 1:3].to_numpy().astype(int)
        pt2 = pt1 + detections.iloc[:, 3:5].to_numpy().round().astype(int)

        return np.hstack((pt1, 
                          pt2, 
                          np.zeros((len(pt1), 1)), # category defaults to 0 for training
                          np.c_[detections.iloc[:, 5].to_numpy()]))
    

    def update_current_tracks(self):
        """ Update tracks for current frame """

        # get all detections at current frame
        detections = self.detections[self.detections.frame == self.frame]

        if not detections.empty:
            detections = self._get_detection_bboxes(detections)
        else:
            detections = np.empty((0, 6))

        # update/associate current tracklets from tracker
        self.current_tracks = self.tracker.update(detections)

        # get ground truth tracks
        self.truth_tracks = self.ground_truth.loc[self.ground_truth.frame == self.frame, :]
        
        # increment frame number
        self.frame += 1


    def associate_gt(self):
        """ Associate Current Tracks to Ground Truth Tracks """

        # get current track bbox for all tracks
        current_track_bboxes = [trk.get_state()[0].astype(int) 
                                for trk in self.current_tracks]
        
        self._update_gt_bboxes()
        
        self.matches, \
        self.false_positives, \
        self.missed_tracks, \
        _                = associate(self.truth_bboxes, 
                                     current_track_bboxes, 
                                     thresh=0.3)
    

    def _get_mismatch_errors(self):
        """ Updates ID map and mismatch errors """
        self.mismatch_errors = 0

        # ground truth IDs
        gt_ids = self.truth_tracks.id.iloc[self.matches[:, 0]].to_numpy()

        # get all track IDs (find more clean way to do this later)
        track_ids = [track.id for track in np.array(self.current_tracks)[self.matches[:, 1]]]

        new_id_map = dict(zip(track_ids, gt_ids))

        # check if any new track has had an ID switch
        for id_ in new_id_map.keys():
            # if ID was previously tracked and it's corresponding ground truth ID changed
            if (id_ in self.id_map.keys()) and (new_id_map[id_] != self.id_map[id_]):
                self.mismatch_errors += 1

            # update id map
            self.id_map = new_id_map


    def update_age_diff(self):
        """ Update total age difference for matched tracks only
            We already penelize the agent for missing tracks and 
            getting false positives
            """
        # reset total age diff
        self.total_age_diff = 0

        # get total age of ground truth tracks
        gt_track_age = 0
        frame_idx = self.ground_truth.frame <= self.frame
        for gt_id in self.id_map.values():
            id_idx = self.ground_truth.id == gt_id
            gt_track_age += sum(frame_idx & id_idx)

        # get total age of current tracks
        curr_track_age = 0
        for track in self.current_tracks:
            curr_track_age += track.age

        self.total_age_diff = abs(gt_track_age - curr_track_age)


    def get_reward(self):
        """ Environment Callback for Computing the rewards """

        # compute cost penalty to enforce good bounding box predictions
        self.cost_penalty = sum([1 - track.observation[12] for track in self.current_tracks])

        # compute rewards
        reward = len(self.false_positives) \
                 + len(self.missed_tracks) \
                 + self.mismatch_errors \
                 + self.cost_penalty # \
                 # + self.total_age_diff

        return -reward
    
    

    def iterate_frame(self):
        """ obtains state/observations at the next frame """

        # get detections and update current tracks for each frame
        self.update_current_tracks()

        # associate tracks with Ground Truth
        self.associate_gt()

        # get ID map and mismatch errors
        # compute number of mismatch errors and update age diff
        if len(self.matches) > 1:
            self._get_mismatch_errors()
            # self.update_age_diff()
        else:
            self.mismatch_errors = self.truth_bboxes.shape[0]
            # self.total_age_diff = 0


    def get_observations(self):
        """ Obtains a vector of all observations for the
            current frame
            """
        # area normalization parameter
        area_norm = self.frame_size[0]*self.frame_size[1] / 4

        observations = {}
        for track in self.current_tracks:
            obs = track.observation

            # normalize obsrvations
            obs[0] /= self.frame_size[1] # xpos
            obs[1] /= self.frame_size[0] # ypos
            obs[2] /= area_norm # area
            obs[4] = sigmoid(obs[4] / self.frame_size[1]) # xvel
            obs[5] = sigmoid(obs[5] / self.frame_size[0]) # yvel
            obs[6] = sigmoid(obs[6] / area_norm) # area vel

            obs[7]  /= self.frame_size[1] # detected xpos
            obs[8]  /= self.frame_size[0] # detected ypos
            obs[9]  /= area_norm # detected area

            # normalize between 0-1
            obs[16] = sigmoid(obs[16] - 3) # frames since last association
            obs[17] = sigmoid(obs[17] - 3) # hit streak

            # track ID maps to observation
            observations.update({track.id : obs})

        return observations
    
    @staticmethod
    def take_action(track, action):
        """ Take action for a single observation vector 

            NOTE: for now assume all other trackfile updates
            occur within the obstacle and tracker classes

            Inputs:
                track - track file object
                action - discrete action to take
            Outputs:
                track - updated trackfile object
            """
        # terminate track
        # possibly use a min_age here to determine if the track can be deleted
        # or pass track age to MARLMOT
        if action == 0:
            # reset track to inactive
            track.track_mode = 0 
            # mark track to be deleted
            track.age = -1

        # restart track with detection (handles motion model failure)
        elif action == 1:
            # reset Kalman filter with new detection
            detection = convert_x_to_bbox(track.detection[0:4]).flatten()
            track.reset_kf(detection)

            # set track to visible
            track.track_mode = 1

        # filter update with prediction and detection
        elif action == 2:
            # perform update with new detection
            detection = convert_x_to_bbox(track.detection[0:4]).flatten()
            track.update(detection)

            # set track to visible
            track.track_mode = 1

        # filter update with prediction only 
        # (track detection is unreliable)
        elif action == 3:
            # no action since prediction has already been made
            # set track to visible
            track.track_mode = 1

        # filter update with prediction only
        # track is placed in a hidden state
        elif action == 4:
            # no action since prediction has already been made
            # set track to hidden
            track.track_mode = 2

        return track


    def take_actions(self, actions):
        """ Take actions for all current tracks/observations
            Inputs:
                actions - dict mapping current track id to discrete action
            """
        updated_tracks = []

        for i in range(len(self.tracker.tracks)):
            track = self.tracker.tracks[i]
            try:
                action = actions[track.id]
                # update track within tracker
                self.tracker.tracks[i] = self.take_action(track, action)
                updated_tracks.append(self.tracker.tracks[i])
            except KeyError:
                continue

        self.current_tracks = updated_tracks


    def step(self, actions):
        """ Generate observations and rewards for a single frame
            Inputs:
                # actions - (Mx1 array) actions for previous observations
                actions - length M dict that maps track ids to discrete actions for previous observations
            Outputs:
                # observations - (Nx18 array) array of (18x1) observation vectors
                observations - length N dict that maps track ids to (18x1) observation vectors
                # rewards - (Nx1) array of rewards
                rewards - length N list of rewards
                done - (Bool) indicates whether the current video is complete
            """
        # implement actions (updates current tracks)
        if len(actions) > 0:
            self.take_actions(actions)

        done = False
        self.iterate_frame()

        # ensure that rewards are reflective of the previous actions
        rewards = (np.repeat(self.get_reward(), len(actions))/len(actions)).tolist()
        observations = self.get_observations()

        # subtract 1 since frame count is incremented in iterate_frame
        # subtract 1 allows for final observations before batch loop exit
        if self.detections.frame.max() == (self.frame - 1):
            done = True

        return observations, rewards, done
    

    def reset(self):
        """ Resets everything """
        self.frame = 0 # current frame index
        self.current_tracks = {} # confirmed tracks
        self.truth_tracks = [] # current truth tracks
        self.truth_bboxes = [] # current truth bboxes
        self.id_map = {} # maps tracker IDs to ground truth IDs
        self.mismatch_errors = 0 # number of current mismatch errors
        self.matches = []
        self.false_positives = []
        self.missed_tracks =[]
        self.cost_penalty = 0
        self.total_age_diff = 0 # total age different between current and GT tracks

        # ensure that the Tracker is reset
        self.tracker.reset()
        observations = {-1 : np.zeros((18, 1))}

        return observations
    

    def get_default_actions(self):
        """ Obtains default action of 2 for each current track """
        actions = {}
        for track in self.current_tracks:
            actions.update({track.id : 2})
        
        return actions 