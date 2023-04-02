"""
    Custom dataloader for training on MOT15 Challenge data
"""

import os
from glob import glob
import configparser
import pandas as pd

# custom dataloader for MOT Challenge data
class TrackDataloader():
    def __init__(self, datafolder, mode="train"):
        """ Custom dataloader for MOT Challenge data
            detection_paths is assumed to always contain matching
            paths for each truth path.
            Args:
                datafolder - (str) folder where MOT15 data is stored
                mode - (str) mode for dataloader (train or test)
            """
        self.mode = mode.lower()
        
        # get data
        train_names = next(iter(os.walk(datafolder)))[1]

        # get individual folders of each video
        self.data_paths = []
        for name in train_names:
            self.data_paths.append(os.path.join(datafolder, name))


        # store current ground truth and detection folder name
        self.current_video = ""

        self.track_cols = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "valid"]
        self.detect_cols = ["frame", "bb_left", "bb_top", "bb_width", "bb_height", "conf"]
    

    def get_gt_tracks(self, data_folder):
        """ Obtains ground truth tracks DataFrame from input train folder.
            Ground Truth DataFrame contains all ground truth bounding boxes
            and ids for every frame
            Inputs:
                data_folder - train folder path
                gt_cols - Desired column names for ground truth DataFrame
            Outputs:
                ground_truth_tracks - Ground Truth Tracks DataFrame 
            """
        ground_truth_tracks = pd.read_csv(os.path.join(data_folder, "gt/gt.txt"), 
                                          usecols=[0,1,2,3,4,5,6], 
                                          header=None)
        # set default column names
        ground_truth_tracks.columns = self.track_cols

        # remove invalid ground truth tracks 
        ground_truth_tracks = ground_truth_tracks[ground_truth_tracks["valid"] == 1].drop(columns=["valid"])

        return ground_truth_tracks
    

    def get_gt_detections(self, data_folder):
        """ Obtains ground truth Detections DataFrame from input train folder.
            Ground Truth DataFrame contains all ground truth detection bounding boxes
            and confidence score for every frame. Occluded objects are not included.
            Inputs:
                data_folder - train folder path
            Outputs:
                detections - Ground Truth Tracks DataFrame 
            """
        detections = pd.read_csv(os.path.join(data_folder, "det/det.txt"), 
                                 usecols=[0,2,3,4,5,6], 
                                 header=None)

        detections.columns = self.detect_cols

        # scale confidence to 0-1
        detections.conf = (detections.conf - detections.conf.min()) \
                          / (detections.conf.max() - detections.conf.min())

        return detections
    

    @staticmethod
    def get_frame_size(data_folder):
        """ Obtains frame size for current video 
            Inputs:
                data_folder - train folder path
            Outputs:
                frame_size (num rows, num cols)
            """
        config = configparser.ConfigParser()
        config.read(os.path.join(data_folder, "seqinfo.ini"))
        frame_size = (int(config.get("Sequence", "imHeight")), # num rows 
                      int(config.get("Sequence", "imWidth")))  # num cols
        return frame_size
    

    @staticmethod
    def get_frame_rate(data_folder):
        """ Obtains frame size for current video 
            Inputs:
                data_folder - train folder path
            Outputs:
                frame_rate
            """
        config = configparser.ConfigParser()
        config.read(os.path.join(data_folder, "seqinfo.ini"))

        return int(config.get("Sequence", "frameRate"))
    
    @staticmethod
    def get_frame_paths(data_folder):
        """ Returns filepaths to image frames """
        return glob(os.path.join(data_folder, "img1/*.jpg"))


    def __getitem__(self, idx):
        """ Obtains data for a given index
            Inputs: 
                idx - current index
            Outputs:
                ground_truth - DataFrame of Ground Truth tracks at every frame
                detections - DataFrame of detections at every frame
                frame_size - frame size list (num rows, num cols)
            """
        data_folder = self.data_paths[idx]
        train_name = os.path.basename(data_folder)

        if self.mode == "train":
            ground_truth = self.get_gt_tracks(data_folder)
        else:
            ground_truth = None

        detections = self.get_gt_detections(data_folder)
        frame_size = self.get_frame_size(data_folder)

        # store current ground truth and video names 
        self.current_video = train_name

        return ground_truth, detections, frame_size
    
    def __len__(self):
        return len(self.data_paths)
