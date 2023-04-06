"""
    This script either evaluates a given policy or obtains the results 
    of running the SORT algorithm on a given set of detections. This 
    script is intended to evaluate the results on all videos in the
    MOT15 training split.
"""
import os
import argparse
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from train_world import TrainWorld
from dataloader import TrackDataloader
from network import Net
from ppo import PPO
from track_utils import *

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def get_args():
    """
        Parses arguments from command line.
        Outputs:
            args - the arguments parsed
    """
    parser = argparse.ArgumentParser()

    # set default paths here
    parser.add_argument("--policy", dest="policy", type=str,
                        default=os.path.join(DIR_PATH, "trained_models\actor_1161.pth"))
    parser.add_argument("--datafolder", dest="datafolder", type=str, 
                        default=r"C:\Users\itber\Documents\datasets\MOT15\test")
    parser.add_argument("--mode", dest="mode", type=str, default="marlmot")
    parser.add_argument("--iou_threshold", dest="iou_threshold", type=float, default=0.3)
    parser.add_argument("--min_age", dest="min_age", type=int, default=1)
    parser.add_argument("--device", dest="device", type=str, default=r"cpu") 
    args = parser.parse_args()

    return args


def get_sort_actions(observations):
        """
            Obtains actions and logprobs from current observations for SORT policy
            i.e. action is always equal to 3
            Inputs:
                observations - (dict) maps track ids to (18x1) observation vectors
                device - (str) device to use
            Outputs:
                actions - (dict) maps track ids to discrete actions for observations
                logprobs -- (tesnor) log probabilities of each action
        """
        # handle initial frame where no observations are made
        if len(observations) == 0:
            return {}, []

        # get default SORT action
        actions = torch.ones((len(observations),)) * 3

        # get logprob of each SORT action
        logprobs = torch.ones_like(actions)

        # map track IDs to actions
        try:
            actions = dict(zip(observations.keys(), 
                               actions.cpu().numpy()))
        except TypeError:
            # handle case for length 1 observation
            actions = dict(zip(observations.keys(), 
                               [actions.cpu().numpy().tolist()]))
            logprobs = logprobs.unsqueeze(0)
        
        return actions, logprobs


def get_sort_rollout(dataloader, iou_threshold, min_age):
    """ Shameless near copy of PPO code to compute SORT rollout """
    batch_obs = []
    batch_actions = []
    batch_logprobs = []
    batch_rewards = []

    # store metrics
    num_false_positives = 0
    num_false_negatives = 0
    num_mismatch_errrors = 0
    cost_penalties = 0
    total_num_tracks = 0

    for (ground_truth, detections, frame_size) in dataloader:
        
        # initialize world object to collect rollouts
        tracker = HungarianTracker(iou_threshold=iou_threshold, 
                                   min_age=min_age)
        world = TrainWorld(tracker=tracker, 
                           ground_truth=ground_truth, 
                           detections=detections,
                           frame_size=frame_size)

        # initialize episode rewards list
        ep_rewards = []

        # accumulate total number of tracks
        total_num_tracks += len(ground_truth)

        # take initial step to get first observations
        observations, _, _ = world.step({})

        # collect (S, A, R) trajectory for entire video
        while True:    

            # append observations first
            batch_obs += list(observations.values())

            # take actions
            actions, logprobs = get_sort_actions(observations)
            # get rewards and new observations
            observations, rewards, done = world.step(actions)

            # get metrics
            num_false_positives += len(world.false_positives)
            num_false_negatives += len(world.missed_tracks)
            num_mismatch_errrors += world.mismatch_errors
            cost_penalties += world.cost_penalty

            # store actions and new rewards 
            batch_rewards.append(rewards)
            batch_actions += list(actions.values())
            batch_logprobs += logprobs

            # assume that tracks at each frame occur at the same time step
            ep_rewards.append(rewards) 

            if done:
                break

    mota = 1 - ((num_false_positives 
                     + num_false_negatives 
                     + num_mismatch_errrors)) / total_num_tracks

    metrics = (len(batch_obs), 
               num_false_positives, 
               num_false_negatives, 
               num_mismatch_errrors, 
               cost_penalties,
               mota)

    return metrics


def eval_sort(dataloader, iou_threshold, min_age):
    """ Special function to evaluate the results of SORT on a given dataset """
    print("Obtaining SORT batch rollouts...")

    batch_len, \
    false_positives, \
    false_negatives, \
    mismatch_errrors, \
    cost_penalty, \
    mota = get_sort_rollout(dataloader, 
                            iou_threshold, 
                            min_age)
    
    # display metrics
    print("batch length: ", batch_len)
    print("false positives: ", false_positives)
    print("false negatives: ", false_negatives)
    print("mismatch errrors: ", mismatch_errrors)
    print("cost penalty: ", cost_penalty.round(4).squeeze())
    print("total: ", false_positives 
                     + false_negatives 
                     + mismatch_errrors 
                     + cost_penalty.round(4).squeeze())
    print("MOTA: ", mota)


def eval_marlmot(dataloader, policy_path, iou_threshold, min_age):
    """ Evaluates MARLMOT policy """
    # get actor/policy
    policy = Net(input_dim=18, output_dim=5).to(device)
    policy.load_state_dict(torch.load(policy_path))
    policy.eval();

    # get default PPO class
    ppo = PPO(dataloader, TrainWorld, Net, epochs=1, 
              iou_threshold=iou_threshold, min_age=min_age, 
              device=device)
              
    # set PPO actor to current actor/policy
    ppo.actor = policy

    # compute a single batch on all data
    print("Obtaining Batch rollouts...")
    batch_obs, _, _, _ = ppo.batch_rollout()

    # display metrics
    false_positives = ppo.metrics["false_positives"][0]
    false_negatives = ppo.metrics["false_negatives"][0]
    mismatch_errrors = ppo.metrics["mismatch_errrors"][0]
    cost_penalty = ppo.metrics["cost_penalty"][0].round(4).squeeze()
    mota = ppo.metrics["mota"][0]

    print("batch length: ", len(batch_obs))
    print("action ratios: ", np.array(ppo.metrics["action_ratios"]).round(4).squeeze())
    print("false positives: ", false_positives)
    print("false negatives: ", false_negatives)
    print("mismatch errrors: ", mismatch_errrors)
    print("cost penalty: ", cost_penalty)
    print("total: ", false_positives 
                     + false_negatives 
                     + mismatch_errrors 
                     + cost_penalty)
    print("MOTA: ", mota)


if __name__ == "__main__":

    # parse arguments
    args = get_args()
    policy_path = args.policy
    datafolder = args.datafolder
    mode = args.mode.lower()
    iou_threshold = args.iou_threshold
    min_age = args.min_age
    device = args.device 

    # get dataloader
    dataloader = TrackDataloader(datafolder)

    if mode == "marlmot":
        print("Evaluating MARLMOT")
        eval_marlmot(dataloader, policy_path, iou_threshold, min_age)
    else:
        print("Evaluating SORT")
        eval_sort(dataloader, iou_threshold, min_age)
    
